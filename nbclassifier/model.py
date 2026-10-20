"""Sparse multinomial naive Bayes over a category hierarchy.

SparseMultinomialNB
    A drop-in for sklearn's MultinomialNB that never densifies the
    (classes x features) count matrix. sklearn stores feature_count_ and
    feature_log_prob_ as dense C x F arrays (3,600 classes x 200,000 features
    = 5.8 GB in float64). Here the counts stay sparse and the log-likelihood
    is computed with an exact algebraic rewrite:

        log(fc + a) - log_norm  =  log(a) - log_norm  +  log1p(fc / a)

    The last term is zero wherever fc is zero, so it is sparse. Hence

        jll = X @ log1p(FC/a).T  +  rowsum(X) * (log a - log_norm)  +  log_prior

    which equals sklearn's _joint_log_likelihood up to floating-point
    rounding (tests/test_model.py checks this against sklearn itself).
    Counts do not depend on alpha or fit_prior, so one fit can be re-used
    for any (alpha, fit_prior) - the grid search exploits this.

HierarchicalNB
    One SparseMultinomialNB per hierarchy level, fitted on *path prefixes*
    (so a name repeated under two parents is never merged). Only category
    paths seen in training are ever predicted. Three prediction modes:

    independent  P(path) = prod_d P_d(prefix_d | x), each level's model
                 normalised over all classes of that level. Raw product;
                 sums to less than 1 over the valid paths.
    conditional  P(path) = prod_d P(prefix_d | prefix_{d-1}, x): the chain
                 rule with one naive Bayes per parent node. This is computed
                 exactly from the level models: a per-parent MultinomialNB
                 has the same feature counts for its children as the level
                 model, and its prior differs only by a constant within each
                 sibling group, which a softmax over the siblings removes.
                 Sums to 1 over valid paths.
    flattened    One model over full paths (level-4 categories); the levels
                 above are implied by the only valid path. Sums to 1.

    Confidence at depth d is the summed score of every valid path sharing
    the predicted path's first d levels (for 'conditional' this is exactly
    prod_{k<=d} P(prefix_k | parent)). Margin at depth d is that confidence
    minus the best competing depth-d prefix (negative when another prefix
    holds more mass than the one on the predicted path). No calibration is
    applied anywhere: these are raw model scores.
"""

from __future__ import annotations

import copy
import gc
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import HyperParams

DEFAULT_BATCH_SIZE = 2000


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _log_softmax(s: np.ndarray) -> np.ndarray:
    m = s.max(axis=1, keepdims=True)
    m[~np.isfinite(m)] = 0.0
    z = s - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _segmented_log_softmax(s: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """log-softmax within contiguous column blocks (sibling groups)."""
    m = np.maximum.reduceat(s, starts, axis=1)
    m[~np.isfinite(m)] = 0.0
    z = s - np.repeat(m, sizes, axis=1)
    lse = np.log(np.add.reduceat(np.exp(z), starts, axis=1))
    return z - np.repeat(lse, sizes, axis=1)


def _as_csr(X) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr().astype(np.float64, copy=False)
    return sp.csr_matrix(np.asarray(X, dtype=np.float64))


# ---------------------------------------------------------------------------
# Sparse multinomial naive Bayes
# ---------------------------------------------------------------------------

class SparseMultinomialNB:
    """Multinomial naive Bayes with sparse class-feature counts.

    Classes are the integers 0..n_classes-1 (callers encode labels).
    Mirrors sklearn.naive_bayes.MultinomialNB(alpha, fit_prior, force_alpha=True).
    """

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self._cache = None

    # -- fitting ------------------------------------------------------------
    def fit(self, X, y: np.ndarray, n_classes: Optional[int] = None) -> "SparseMultinomialNB":
        X = _as_csr(X)
        if X.nnz and X.data.min() < 0:
            raise ValueError("Negative values in X; multinomial NB needs non-negative features")
        y = np.asarray(y, dtype=np.int64)
        n = X.shape[0]
        C = int(n_classes if n_classes is not None else y.max() + 1)
        Y = sp.csr_matrix((np.ones(n), (np.arange(n), y)), shape=(n, C))
        fc = (Y.T @ X).tocsr()
        fc.sum_duplicates()
        fc.eliminate_zeros()
        self.feature_count_ = fc                                   # C x F, sparse
        self.class_count_ = np.bincount(y, minlength=C).astype(np.float64)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = C
        self._cache = None
        return self

    def with_params(self, alpha: float, fit_prior: bool) -> "SparseMultinomialNB":
        """A copy sharing the fitted counts, with different alpha / fit_prior."""
        new = copy.copy(self)
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        new.alpha, new.fit_prior, new._cache = float(alpha), bool(fit_prior), None
        return new

    # -- derived parameters ---------------------------------------------------
    def _params(self):
        if self._cache is None:
            a, F = self.alpha, self.n_features_in_
            fc = self.feature_count_
            total = np.asarray(fc.sum(axis=1)).ravel()
            log_norm = np.log(total + a * F)
            S = fc.copy()
            S.data = np.log1p(S.data / a)
            S_T = S.T.tocsr()
            const = np.log(a) - log_norm
            if self.fit_prior:
                with np.errstate(divide="ignore"):
                    prior = np.log(self.class_count_) - np.log(self.class_count_.sum())
            else:
                prior = np.full(self.n_classes_, -np.log(self.n_classes_))
            self._cache = (S_T, const, prior)
        return self._cache

    @property
    def class_log_prior_(self) -> np.ndarray:
        return self._params()[2]

    @property
    def feature_log_prob_(self) -> np.ndarray:
        """Dense C x F matrix, for testing/inspection only (can be very large)."""
        a, F = self.alpha, self.n_features_in_
        total = np.asarray(self.feature_count_.sum(axis=1)).ravel()
        return np.log(self.feature_count_.toarray() + a) - np.log(total + a * F)[:, None]

    # -- inference ------------------------------------------------------------
    def joint_log_likelihood(self, X) -> np.ndarray:
        S_T, const, prior = self._params()
        X = _as_csr(X)
        rs = np.asarray(X.sum(axis=1)).ravel()
        jll = (X @ S_T).toarray()
        jll += rs[:, None] * const[None, :]
        jll += prior[None, :]
        return jll

    def predict_log_proba(self, X) -> np.ndarray:
        return _log_softmax(self.joint_log_likelihood(X))

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        return self.joint_log_likelihood(X).argmax(axis=1)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = None
        return state


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

class HierarchicalNB:
    """Naive Bayes over a fixed-depth category hierarchy (see module docstring)."""

    def __init__(self, level_columns: Sequence[str], hp: HyperParams):
        self.level_columns = list(level_columns)
        self.hp = hp
        self.vectorizer = None
        self.level_models: List[SparseMultinomialNB] = []
        self.classes_: List[pd.DataFrame] = []   # per depth: columns parent (int), label (str)
        self.path_codes_: List[np.ndarray] = []  # per depth: prefix code of every valid path
        self.fitted_ = False

    @property
    def n_levels(self) -> int:
        return len(self.level_columns)

    # -- fitting ------------------------------------------------------------
    def fit(self, texts: pd.Series, labels: pd.DataFrame, vectorizer=None, X=None) -> "HierarchicalNB":
        """Fit on cleaned texts and a labels frame holding the level columns.

        Pass an already-fitted `vectorizer` and its matrix `X` to skip
        vectorizing (the grid search does this to share work).
        """
        if X is None:
            from .vectorizers import build_vectorizer

            self.vectorizer = build_vectorizer(self.hp)
            X = self.vectorizer.fit_transform(pd.Series(texts).fillna("").astype(str))
        else:
            self.vectorizer = vectorizer
        row_codes = self._build_index(labels)
        X = _as_csr(X)
        self.level_models = []
        for d in range(self.n_levels):
            m = SparseMultinomialNB(self.hp.alpha, self.hp.fit_prior)
            m.fit(X, row_codes[d], n_classes=len(self.classes_[d]))
            self.level_models.append(m)
        del row_codes
        gc.collect()
        self.fitted_ = True
        return self

    def _build_index(self, labels: pd.DataFrame) -> List[np.ndarray]:
        lab = labels[self.level_columns].astype(str).reset_index(drop=True)
        parent = np.zeros(len(lab), dtype=np.int64)
        self.classes_, row_codes = [], []
        for col in self.level_columns:
            pairs = pd.DataFrame({"parent": parent, "label": lab[col].to_numpy()})
            uniq = (pairs.drop_duplicates()
                         .sort_values(["parent", "label"], kind="mergesort")
                         .reset_index(drop=True))
            uniq["code"] = np.arange(len(uniq), dtype=np.int64)
            codes = pairs.merge(uniq, on=["parent", "label"], how="left")["code"].to_numpy()
            self.classes_.append(uniq[["parent", "label"]])
            row_codes.append(codes)
            parent = codes
        # Every deepest-level class is exactly one valid path.
        P = len(self.classes_[-1])
        codes = [None] * self.n_levels
        codes[-1] = np.arange(P, dtype=np.int64)
        for d in range(self.n_levels - 1, 0, -1):
            codes[d - 1] = self.classes_[d]["parent"].to_numpy()[codes[d]]
        self.path_codes_ = codes
        self._derived = None
        return row_codes

    def clone_with(self, alpha: float, fit_prior: bool, prediction_mode: str) -> "HierarchicalNB":
        """Same vocabulary and counts, different NB settings (cheap)."""
        new = copy.copy(self)
        new.hp = HyperParams.from_dict({**self.hp.to_dict(), "alpha": alpha,
                                        "fit_prior": fit_prior, "prediction_mode": prediction_mode})
        new.level_models = [m.with_params(alpha, fit_prior) for m in self.level_models]
        new._derived = None
        return new

    # -- structure helpers ----------------------------------------------------
    def _structure(self):
        if getattr(self, "_derived", None) is None:
            P = self.n_paths
            groups, masks = [], []
            for d in range(self.n_levels):
                parent = self.classes_[d]["parent"].to_numpy()
                starts = np.flatnonzero(np.r_[True, parent[1:] != parent[:-1]])
                sizes = np.diff(np.r_[starts, len(parent)])
                groups.append((starts, sizes))
                C = len(parent)
                masks.append(sp.csr_matrix((np.ones(P), (np.arange(P), self.path_codes_[d])), shape=(P, C)))
            labels = [self.classes_[d]["label"].to_numpy()[self.path_codes_[d]] for d in range(self.n_levels)]
            self._derived = {"groups": groups, "masks": masks, "path_labels": labels}
        return self._derived

    @property
    def n_paths(self) -> int:
        return len(self.classes_[-1])

    @property
    def n_features(self) -> int:
        return self.level_models[0].n_features_in_ if self.level_models else 0

    @property
    def classes_per_level(self) -> Dict[str, int]:
        return {c: len(k) for c, k in zip(self.level_columns, self.classes_)}

    def valid_paths(self) -> pd.DataFrame:
        """One row per valid path, label at every level."""
        labels = self._structure()["path_labels"]
        return pd.DataFrame({c: labels[d] for d, c in enumerate(self.level_columns)})

    # -- inference ------------------------------------------------------------
    def transform(self, texts) -> sp.csr_matrix:
        return self.vectorizer.transform(pd.Series(texts).fillna("").astype(str))

    def joint_log(self, X) -> np.ndarray:
        """log score of every valid path (rows x n_paths)."""
        mode = self.hp.prediction_mode
        st = self._structure()
        if mode == "flattened":
            return self.level_models[-1].predict_log_proba(X)
        joint = None
        for d, m in enumerate(self.level_models):
            jll = m.joint_log_likelihood(X)
            if mode == "conditional":
                starts, sizes = st["groups"][d]
                lp = _segmented_log_softmax(jll, starts, sizes)
            else:  # independent
                lp = _log_softmax(jll)
            contrib = lp[:, self.path_codes_[d]]
            joint = contrib if joint is None else joint + contrib
            del jll, lp
        return joint

    def predict_arrays(self, X, batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(best path index, confidence n x L, margin n x L), batched to bound memory."""
        X = _as_csr(X)
        n, L = X.shape[0], self.n_levels
        st = self._structure()
        best = np.empty(n, dtype=np.int64)
        conf = np.empty((n, L), dtype=np.float64)
        margin = np.empty((n, L), dtype=np.float64)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            joint = self.joint_log(X[s:e])
            b = joint.argmax(axis=1)
            scores = np.exp(joint)
            del joint
            rows = np.arange(e - s)
            for d in range(L):
                # (P x C_d indicator).T @ scores.T -> summed score of each depth-d prefix
                mass = np.ascontiguousarray(np.asarray(st["masks"][d].T @ scores.T).T)
                chosen = self.path_codes_[d][b]
                c = mass[rows, chosen]
                if mass.shape[1] > 1:
                    mass[rows, chosen] = -np.inf
                    other = mass.max(axis=1)
                else:
                    other = np.zeros_like(c)
                conf[s:e, d] = c
                margin[s:e, d] = c - other
            best[s:e] = b
            del scores
        return best, conf, margin

    def predict_frame(self, texts=None, X=None, batch_size: int = DEFAULT_BATCH_SIZE) -> pd.DataFrame:
        """Predicted label per level plus L{d} Confidence and L{d} Margin."""
        if X is None:
            X = self.transform(texts)
        best, conf, margin = self.predict_arrays(X, batch_size=batch_size)
        labels = self._structure()["path_labels"]
        out = {}
        for d, col in enumerate(self.level_columns):
            out[f"Predicted {col}"] = labels[d][best]
        for d in range(self.n_levels):
            out[f"L{d + 1} Confidence"] = conf[:, d]
        for d in range(self.n_levels):
            out[f"L{d + 1} Margin"] = margin[:, d]
        return pd.DataFrame(out)

    # -- input quality ---------------------------------------------------------
    def input_quality(self, texts) -> pd.DataFrame:
        """n_tokens, n_known and oov_rate of each text against the vocabulary.

        Tokens are unigrams after the vectorizer's own preprocessing,
        tokenisation and stop-word removal; 'known' means the unigram is a
        feature of the fitted vectorizer.
        """
        vec = self.vectorizer
        pre, tok = vec.build_preprocessor(), vec.build_tokenizer()
        stop = vec.get_stop_words() or frozenset()
        vocab = vec.vocabulary_
        n_tok, n_known = [], []
        for doc in pd.Series(texts).fillna("").astype(str):
            toks = [t for t in tok(pre(doc)) if t not in stop]
            n_tok.append(len(toks))
            n_known.append(sum(1 for t in toks if t in vocab))
        n_tok = np.asarray(n_tok)
        n_known = np.asarray(n_known)
        with np.errstate(divide="ignore", invalid="ignore"):
            oov = np.where(n_tok > 0, 1.0 - n_known / np.maximum(n_tok, 1), 1.0)
        return pd.DataFrame({"n_tokens": n_tok, "n_known": n_known, "oov_rate": oov})

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_derived"] = None
        return state
