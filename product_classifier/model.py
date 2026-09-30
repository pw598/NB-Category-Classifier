"""The classifier: MultinomialNB models combined into a hierarchy prediction.

Three modes, two of which score whole paths and share all the machinery
below _joint_log_from_batch:

    joint_path        one model per level, trained on everything, and the
                      per-level marginals multiplied together.
    conditional_path  one model per parent node, each trained only on the
                      rows beneath it, combined by the chain rule.
    per_level         each level's own argmax, no path constraint.

The two path modes differ only in how the (rows x n_paths) log-score
matrix is built. Everything downstream -- prefix sums, per-depth
confidence, the cascade, refusals -- is shared.
"""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from .config import EPS, ModelConfig, NBConfig, PredictionMode, VectorizerConfig

PROBABILITY_COL = "Probability"
EVIDENCE_COL = "Path Evidence"


class HierarchicalPathClassifier:
    """Trains one MultinomialNB per level and predicts either a whole
    hierarchy path (joint) or each level on its own (per-level).

    The two modes share the same fitted models, so a single fit supports
    both and they can be compared directly.

    Memory: the design keeps four things bounded --
      1. the document-term matrix stays sparse end to end;
      2. only hierarchy paths actually observed in training are scored,
         never the full L1 x L2 x L3 x L4 cross product;
      3. inference runs in batches, so peak memory is independent of
         test-set size;
      4. probabilities are combined in log space (a sum, not a product) in
         float32, which avoids underflow and halves the score matrix.
    """

    def __init__(
        self,
        level_columns: List[str],
        config: Optional[ModelConfig] = None,
    ):
        self.config = config or ModelConfig()
        n = self.config.n_levels
        if not 1 <= n <= len(level_columns):
            raise ValueError(
                f"n_levels={n} is outside the {len(level_columns)} available levels."
            )
        self.level_columns = list(level_columns[:n])
        self.dtype = np.dtype(self.config.dtype)

        self.vectorizer = None
        self.models: Dict[str, MultinomialNB] = {}
        self.label_to_col: Dict[str, Dict[str, int]] = {}
        self.valid_paths_: Optional[pd.DataFrame] = None
        self._path_idx_arrays: Optional[Dict[str, np.ndarray]] = None
        # conditional_path only: one stacked matrix per depth. See
        # _fit_node_models for the layout.
        self.node_stack_: Optional[Dict[int, dict]] = None
        self.fitted_ = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, texts: pd.Series, labels_df: pd.DataFrame) -> "HierarchicalPathClassifier":
        t0 = time.time()
        self.vectorizer = self.config.vectorizer.build()
        X = self.vectorizer.fit_transform(texts)
        print(
            f"[fit] vectorized {X.shape[0]:,} rows into {X.shape[1]:,} features "
            f"(sparse, nnz={X.nnz:,}) in {time.time() - t0:.1f}s"
        )

        conditional = self.config.prediction_mode == "conditional_path"
        # getattr: a ModelConfig unpickled from a bundle saved before these
        # fields existed will not carry them.
        flat = getattr(self.config, "fit_flat_levels", True) or not conditional

        if flat:
            for level in self.level_columns:
                t1 = time.time()
                y = labels_df[level].astype(str).values
                model = self._fit_level(X, y)
                self.models[level] = model
                self.label_to_col[level] = {
                    lbl: i for i, lbl in enumerate(model.classes_)
                }
                print(
                    f"[fit] trained '{level}': {len(model.classes_):,} classes "
                    f"in {time.time() - t1:.1f}s"
                )
        else:
            print("[fit] fit_flat_levels=False: skipping the per-level models, "
                  "so joint_path and per_level are unavailable on this object")

        self._build_path_index(labels_df, flat=flat)

        if conditional:
            self._fit_node_models(X, labels_df)

        del X
        gc.collect()
        self.fitted_ = True
        return self

    def _fit_level(self, X, y: np.ndarray) -> MultinomialNB:
        """Fit one level's model, in row chunks when configured.

        sklearn's MultinomialNB.fit binarizes the labels into a dense
        (n_rows x n_classes) float64 array. That array is fine at 23 or 148
        classes and enormous at a few thousand, which is what makes the
        deep levels fall over on a large golden set.

        partial_fit accumulates class_count_ and feature_count_ chunk by
        chunk and recomputes the log-probabilities from those totals, so
        the fitted model is identical to the one a single fit() call
        produces -- the counts summed are the same counts, just added in a
        different order. Every row is still used exactly once. Passing the
        full class list up front is what makes this exact rather than an
        approximation: the model allocates for all classes immediately, so
        a chunk that happens to contain no examples of some rare class
        still leaves room for it.
        """
        model = self.config.nb.build()
        chunk = self.config.fit_chunk_size
        n = X.shape[0]

        if not chunk or chunk >= n:
            model.fit(X, y)
            return model

        classes = np.unique(y)
        n_chunks = (n + chunk - 1) // chunk
        print(
            f"[fit]   chunked fit: {n:,} rows in {n_chunks} chunk(s) of "
            f"{chunk:,}, {len(classes):,} classes"
        )
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            model.partial_fit(X[start:end], y[start:end], classes=classes)
        return model

    def _build_path_index(self, labels_df: pd.DataFrame, flat: bool = True) -> None:
        self.valid_paths_ = (
            labels_df[self.level_columns]
            .astype(str)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        n_valid = len(self.valid_paths_)
        if not flat:
            # The path index maps a path to a column of each flat model's
            # predict_proba output. Without those models there is nothing
            # to index into.
            self._path_idx_arrays = None
            print(f"[fit] {n_valid:,} paths observed in training")
            return

        self._path_idx_arrays = {
            level: self.valid_paths_[level].map(self.label_to_col[level]).to_numpy()
            for level in self.level_columns
        }
        n_full = 1
        for level in self.level_columns:
            n_full *= len(self.models[level].classes_)
        print(
            f"[fit] {n_valid:,} paths observed in training vs. {n_full:,} possible "
            f"combinations -> scoring {n_valid / n_full:.6%} of the cross product"
        )

    # ------------------------------------------------------------------
    # conditional_path: one model per parent node
    # ------------------------------------------------------------------

    @staticmethod
    def _group_bounds(keys: np.ndarray):
        """Row positions grouped by key, from one stable sort.

        Returns (order, sorted_keys, starts, bounds) where rows of group i
        are order[bounds[i]:bounds[i + 1]].
        """
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        bounds = np.r_[starts, len(sorted_keys)]
        return order, sorted_keys, starts, bounds

    def _parent_keys(self, frame: pd.DataFrame, d: int) -> np.ndarray:
        """Each row's parent-prefix key at depth d. Depth 1 has one root."""
        parents = self.level_columns[: d - 1]
        if not parents:
            return np.full(len(frame), "", dtype="<U1")
        return frame[parents].astype(str).agg("\x00".join, axis=1).to_numpy()

    def _fit_node_models(self, X, labels_df: pd.DataFrame) -> None:
        """One MultinomialNB per parent, stacked into a matrix per depth.

        Each parent's model is fitted on its own rows alone and predicts
        among its own children, which is what makes the factor conditional
        rather than marginal. The models are then discarded: all scoring
        needs is their feature_log_prob_ and class_log_prior_, and holding
        one stacked matrix per depth beats holding a thousand estimators.

        Layout per depth, in self.node_stack_[d]:
            W             (n_features x C_d) log-probabilities, sibling
                          groups laid out in contiguous column blocks
            b             (C_d,) class log-priors, same column order
            group_starts  first column of each block, for reduceat
            group_sizes   width of each block
            path_col      for each valid path, its column in W, or -1
            fallback      for each valid path, the log-probability to add
                          when its parent has no model

        A parent is left unmodelled when it has one child (the factor is
        then log 1) or fewer rows than min_node_rows. Unmodelled parents
        spread their mass evenly over their children rather than
        contributing nothing, which would silently favour those subtrees.
        """
        levels = self.level_columns
        lab = labels_df[levels].astype(str).reset_index(drop=True)
        paths = self.valid_paths_
        n_paths = len(paths)
        min_rows = max(int(getattr(self.config, "min_node_rows", 0) or 0), 0)

        self.node_stack_ = {}
        for d in range(1, len(levels) + 1):
            t0 = time.time()
            child_col = levels[d - 1]

            row_keys = self._parent_keys(lab, d)
            order, sorted_keys, starts, bounds = self._group_bounds(row_keys)
            child_of_row = lab[child_col].to_numpy()

            path_keys = self._parent_keys(paths, d)
            path_child = paths[child_col].astype(str).to_numpy()

            blocks: List[np.ndarray] = []
            priors: List[np.ndarray] = []
            sizes: List[int] = []
            col_of: Dict[tuple, int] = {}
            next_col = 0
            n_modelled = n_skipped = 0

            for gi in range(len(starts)):
                rows = order[bounds[gi]: bounds[gi + 1]]
                key = sorted_keys[starts[gi]]
                y = child_of_row[rows]
                if len(np.unique(y)) < 2 or len(rows) < min_rows:
                    n_skipped += 1
                    continue
                model = self.config.nb.build()
                model.fit(X[rows], y)
                blocks.append(model.feature_log_prob_.T.astype(self.dtype))
                priors.append(model.class_log_prior_.astype(self.dtype))
                for j, lbl in enumerate(model.classes_):
                    col_of[(key, str(lbl))] = next_col + j
                next_col += len(model.classes_)
                sizes.append(len(model.classes_))
                n_modelled += 1
                del model

            # How many children each parent has, counted over the path table
            # that the score matrix is indexed by.
            n_children: Dict[str, set] = {}
            for k, c in zip(path_keys, path_child):
                n_children.setdefault(k, set()).add(c)

            path_col = np.full(n_paths, -1, dtype=np.int64)
            fallback = np.zeros(n_paths, dtype=np.float64)
            for i in range(n_paths):
                key = path_keys[i]
                col = col_of.get((key, path_child[i]))
                if col is None:
                    fallback[i] = -np.log(max(len(n_children.get(key, ())), 1))
                else:
                    path_col[i] = col

            if blocks:
                W = np.hstack(blocks)
                b = np.concatenate(priors)
                size_arr = np.asarray(sizes, dtype=np.int64)
                gstarts = np.concatenate(
                    ([0], np.cumsum(size_arr)[:-1])
                ).astype(np.int64)
            else:
                W = b = size_arr = gstarts = None
            del blocks, priors

            self.node_stack_[d] = {
                "W": W,
                "b": b,
                "group_starts": gstarts,
                "group_sizes": size_arr,
                "path_col": path_col,
                "fallback": fallback,
            }
            width = 0 if W is None else W.shape[1]
            print(
                f"[fit] depth {d}: {n_modelled:,} parent model(s), "
                f"{n_skipped:,} left uniform, {width:,} stacked columns "
                f"in {time.time() - t0:.1f}s"
            )
            gc.collect()

    @staticmethod
    def _segmented_log_softmax(s, starts, sizes):
        """log-softmax within each contiguous block of columns.

        The max-shift is per block rather than per row: each sibling group
        is its own distribution, and a shared shift would not cancel.
        """
        m_full = np.repeat(np.maximum.reduceat(s, starts, axis=1), sizes, axis=1)
        e = np.exp(s - m_full)
        z = np.add.reduceat(e, starts, axis=1)
        del e
        out = s - m_full - np.repeat(np.log(z), sizes, axis=1)
        del m_full
        return out

    def _conditional_joint_log(self, X_batch) -> np.ndarray:
        """log P(c1|x) + log P(c2|c1,x) + ... for every valid path."""
        from sklearn.utils.extmath import safe_sparse_dot

        joint = np.zeros(
            (X_batch.shape[0], len(self.valid_paths_)), dtype=self.dtype
        )
        for d in sorted(self.node_stack_):
            stack = self.node_stack_[d]
            joint += stack["fallback"].astype(self.dtype)
            if stack["W"] is None:
                continue
            # One matmul for the whole depth, then normalise inside each
            # sibling block -- the same reduceat trick the prefix sums use.
            s = np.asarray(
                safe_sparse_dot(X_batch, stack["W"]), dtype=self.dtype
            ) + stack["b"]
            logp = self._segmented_log_softmax(
                s, stack["group_starts"], stack["group_sizes"]
            )
            del s
            cols = stack["path_col"]
            mask = cols >= 0
            joint[:, mask] += logp[:, cols[mask]]
            del logp
        return joint

    def _flat_joint_log(self, X_batch) -> np.ndarray:
        """log P1(c1|x) + log P2(c2|x) + ... -- marginals, multiplied."""
        joint = np.zeros(
            (X_batch.shape[0], len(self.valid_paths_)), dtype=self.dtype
        )
        for level in self.level_columns:
            proba = np.clip(
                self.models[level].predict_proba(X_batch).astype(self.dtype),
                EPS, 1.0,
            )
            joint += np.log(proba)[:, self._path_idx_arrays[level]]
            del proba
        return joint

    def _joint_log_from_batch(self, X_batch, mode: Optional[str] = None) -> np.ndarray:
        """The (rows x n_paths) log-score matrix every path method works from.

        The only difference between the two path modes lives here, which is
        why nothing downstream -- prefix sums, confidence, thresholds --
        needs to know which one is in use.
        """
        mode = mode or self.config.prediction_mode
        if mode == "conditional_path":
            if not self.node_stack_:
                raise RuntimeError(
                    "conditional_path needs the per-parent models, and this "
                    "classifier was fitted without them. Set "
                    "cfg.model.prediction_mode = 'conditional_path' before fit()."
                )
            return self._conditional_joint_log(X_batch)
        if self._path_idx_arrays is None:
            raise RuntimeError(
                "The per-level models were skipped at fit time "
                "(fit_flat_levels=False), so conditional_path is the only "
                "mode available on this object."
            )
        return self._flat_joint_log(X_batch)

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("Call fit() before predicting.")

    # ------------------------------------------------------------------
    # Batched scoring on a pre-vectorized matrix
    # ------------------------------------------------------------------

    def _iter_batches(self, X):
        n = X.shape[0]
        for start in range(0, n, self.config.batch_size):
            end = min(start + self.config.batch_size, n)
            yield start, end, X[start:end]

    def _predict_per_level_from_X(self, X) -> pd.DataFrame:
        """Independent argmax at each level, plus that level's own probability.

        Levels are not constrained to be consistent with one another, so an
        L4 prediction may not sit under the predicted L1.
        """
        n = X.shape[0]
        out = {}
        for level in self.level_columns:
            model = self.models[level]
            preds = np.empty(n, dtype=object)
            probs = np.empty(n, dtype=np.float64)
            for start, end, X_batch in self._iter_batches(X):
                proba = model.predict_proba(X_batch)
                best = np.argmax(proba, axis=1)
                preds[start:end] = model.classes_[best]
                probs[start:end] = proba[np.arange(end - start), best]
                del proba
            out[level] = preds
            out[f"{level} Probability"] = probs
        result = pd.DataFrame(out)
        # A comparable single-number confidence: product of level confidences.
        prob_cols = [f"{level} Probability" for level in self.level_columns]
        result[PROBABILITY_COL] = result[prob_cols].prod(axis=1)
        return result

    def _resolve_normalise(self, override: Optional[bool], default: bool) -> bool:
        """Explicit argument wins, then ModelConfig, then the method's own
        historical default.

        The three-way resolution exists so that setting one config flag can
        put predict() and predict_prefix_probabilities() on a common scale
        without changing what either does when nobody asks.
        """
        if override is not None:
            return override
        from_cfg = getattr(self.config, "normalise_probabilities", None)
        return default if from_cfg is None else bool(from_cfg)

    def _predict_paths_from_X(
        self, X, normalise: bool = False, mode: Optional[str] = None
    ) -> pd.DataFrame:
        """Best hierarchy path by summed log-probability across levels.

        normalise controls what the reported Probability means:

        False (the default)
            exp of the summed log-probabilities: the raw product of the
            per-level posteriors. Answers "how well does this path fit?",
            and shrinks mechanically with depth, since more numbers below
            one get multiplied.

        True
            that same quantity divided by the total across every valid
            path, so the candidates sum to 1. Answers "what share of the
            belief does this path hold?", and is comparable across depths.
            At full depth this is exactly the figure
            predict_prefix_probabilities() reports for the deepest prefix.
        """
        n = X.shape[0]

        best_path_idx = np.empty(n, dtype=np.int64)
        best_path_logprob = np.empty(n, dtype=self.dtype)
        share = np.empty(n, dtype=np.float64) if normalise else None

        for start, end, X_batch in self._iter_batches(X):
            joint_log = self._joint_log_from_batch(X_batch, mode)
            batch_best = np.argmax(joint_log, axis=1)
            best_path_idx[start:end] = batch_best
            best_path_logprob[start:end] = joint_log[np.arange(end - start), batch_best]
            if normalise:
                # Shift by the row max before exponentiating; the raw joint
                # log-probabilities underflow float64 well before depth 4.
                shifted = joint_log - joint_log.max(axis=1, keepdims=True)
                weights = np.exp(shifted.astype(np.float64))
                share[start:end] = (
                    weights[np.arange(end - start), batch_best] / weights.sum(axis=1)
                )
                del shifted, weights
            del joint_log
            gc.collect()

        result = self.valid_paths_.iloc[best_path_idx].reset_index(drop=True).copy()
        if normalise:
            result[PROBABILITY_COL] = share
            # Keep the absolute figure too: a product unlike anything in the
            # catalogue scores badly against every path, yet normalises to
            # near 1 because the alternatives are equally bad. Normalising
            # alone would hide that.
            result[EVIDENCE_COL] = np.exp(best_path_logprob.astype(np.float64))
        else:
            result[PROBABILITY_COL] = np.exp(best_path_logprob.astype(np.float64))
        return result

    def _top_k_paths_from_X(self, X, k: int) -> pd.DataFrame:
        """The k best paths per row, long-format, for review queues."""
        frames = []
        n_paths = len(self.valid_paths_)
        k = min(k, n_paths)
        for start, end, X_batch in self._iter_batches(X):
            joint_log = self._joint_log_from_batch(X_batch)
            top = np.argpartition(-joint_log, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(end - start)[:, None]
            order = np.argsort(-joint_log[rows, top], axis=1)
            top = top[rows, order]
            for rank in range(k):
                frame = self.valid_paths_.iloc[top[:, rank]].reset_index(drop=True).copy()
                frame[PROBABILITY_COL] = np.exp(
                    joint_log[np.arange(end - start), top[:, rank]].astype(np.float64)
                )
                frame["Rank"] = rank + 1
                frame["Row"] = np.arange(start, end)
                frames.append(frame)
            del joint_log
            gc.collect()
        return pd.concat(frames).sort_values(["Row", "Rank"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API (raw text in)
    # ------------------------------------------------------------------

    def transform(self, texts: pd.Series):
        self._check_fitted()
        return self.vectorizer.transform(texts.fillna("").astype(str))

    def predict(
        self,
        texts: pd.Series,
        mode: Optional[PredictionMode] = None,
        normalise: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Predict with the configured mode (or an override).

        Returns a DataFrame with one column per active level plus a
        'Probability' column.

        normalise decides the scale of that column -- see
        _predict_paths_from_X. Leave it None to take the value from
        ModelConfig.normalise_probabilities, which is itself None by
        default, meaning the raw product. Only joint_path is affected:
        per_level has no set of candidate paths to normalise over, so its
        Probability stays a raw product of per-level confidences, and
        conditional_path already sums to 1 across valid paths, so dividing
        by that total changes nothing.

        conditional_path can only be used if the classifier was fitted with
        it configured -- it needs a different set of models, not a
        different reading of the same ones.
        """
        mode = mode or self.config.prediction_mode
        X = self.transform(texts)
        if mode in ("joint_path", "conditional_path"):
            return self._predict_paths_from_X(
                X, normalise=self._resolve_normalise(normalise, False), mode=mode
            )
        if mode == "per_level":
            if not self.models:
                raise RuntimeError(
                    "per_level needs the flat models, skipped at fit time "
                    "(fit_flat_levels=False)."
                )
            return self._predict_per_level_from_X(X)
        raise ValueError(f"Unknown prediction_mode: {mode!r}")

    def predict_top_k(self, texts: pd.Series, k: int = 3) -> pd.DataFrame:
        """The k highest-probability paths per row. Joint mode only."""
        return self._top_k_paths_from_X(self.transform(texts), k)

    def predict_prefix_probabilities(
        self, texts: pd.Series, normalise: Optional[bool] = None
    ) -> pd.DataFrame:
        """Per-row probability of the best path *prefix* at every depth.

        This is the signal a variable-depth cascade needs, and it is not
        the same thing as the 'Probability' column from predict().

        predict() reports the score of the single best full-depth path: a
        max. If an item is genuinely split across five sibling L4s under
        one confidently-known L2, every individual path scores low and the
        max looks hopeless -- even though the L2 is near-certain. Summing
        the mass over all paths sharing a prefix recovers that, which is
        exactly the case where truncating to L2 is the right call.

        Two further differences from predict(), both deliberate:

        * The values here are normalised over the valid paths, so at each
          depth the prefix probabilities across candidate prefixes sum to
          1 and are directly comparable between depths. predict() reports
          an unnormalised product of per-level posteriors, which shrinks
          mechanically as depth grows. Do not compare the two columns.
        * A 'Margin' is reported per depth: top prefix minus runner-up.
          Often a better safety signal than absolute probability, since
          Naive Bayes is systematically overconfident.

        Returns one row per input, with these columns per depth d:
            L{d} Prefix Probability, L{d} Margin, and the predicted label
            for each level in the winning prefix, as
            'L{d} Predicted {level}'.
        """
        self._check_fitted()
        normalise = self._resolve_normalise(normalise, True)
        X = self.transform(texts)
        n = X.shape[0]
        depths = range(1, len(self.level_columns) + 1)
        groups = self._prefix_groups()

        prob = {d: np.empty(n, dtype=np.float64) for d in depths}
        margin = {d: np.empty(n, dtype=np.float64) for d in depths}
        winner = {d: np.empty(n, dtype=np.int64) for d in depths}

        for start, end, X_batch in self._iter_batches(X):
            joint_log = self._joint_log_from_batch(X_batch)

            # Shift by the row max before exponentiating: the raw joint log
            # probabilities underflow float64 well before depth 4.
            row_max = joint_log.max(axis=1, keepdims=True)
            shifted = joint_log - row_max
            weights = np.exp(shifted.astype(np.float64))
            # Normalised: divide by the total across candidates, so prefixes
            # sum to 1. Raw: undo the max-shift instead, giving the same
            # summed products predict() reports without normalisation.
            total = (
                weights.sum(axis=1, keepdims=True)
                if normalise
                else np.exp(-row_max.astype(np.float64))
            )
            del shifted, joint_log, row_max

            for d in depths:
                order, starts = groups[d]["order"], groups[d]["starts"]
                # Paths pre-sorted by prefix, so contiguous blocks sum with
                # one reduceat rather than a scatter-add per group.
                summed = np.add.reduceat(weights[:, order], starts, axis=1)
                summed /= total
                best = np.argmax(summed, axis=1)
                rows = np.arange(end - start)
                top1 = summed[rows, best]
                if summed.shape[1] > 1:
                    summed[rows, best] = -np.inf
                    top2 = summed.max(axis=1)
                else:
                    top2 = np.zeros_like(top1)
                prob[d][start:end] = top1
                margin[d][start:end] = top1 - top2
                winner[d][start:end] = best
                del summed
            del weights, total
            gc.collect()

        out = {}
        for d in depths:
            labels = groups[d]["labels"]
            for i, level in enumerate(self.level_columns[:d]):
                out[f"L{d} Predicted {level}"] = labels[i][winner[d]]
            out[f"L{d} Prefix Probability"] = prob[d]
            out[f"L{d} Margin"] = margin[d]
        return pd.DataFrame(out)

    def _prefix_groups(self) -> Dict[int, dict]:
        """Path-to-prefix grouping, one entry per depth. Built once, cached.

        Cached lazily rather than in __init__ so that classifiers pickled
        before this method existed still load and work.
        """
        cache = getattr(self, "_prefix_cache", None)
        if cache is not None:
            return cache

        cache = {}
        paths = self.valid_paths_
        for d in range(1, len(self.level_columns) + 1):
            cols = self.level_columns[:d]
            key = paths[cols].astype(str).agg("\x00".join, axis=1).to_numpy()
            order = np.argsort(key, kind="stable")
            sorted_key = key[order]
            # First index of each distinct prefix in the sorted ordering.
            starts = np.flatnonzero(
                np.r_[True, sorted_key[1:] != sorted_key[:-1]]
            )
            first_rows = order[starts]
            cache[d] = {
                "order": order,
                "starts": starts,
                "labels": [paths[c].astype(str).to_numpy()[first_rows] for c in cols],
            }
        self._prefix_cache = cache
        return cache

    def predict_single_level(self, texts: pd.Series, level: str) -> np.ndarray:
        """One level's own argmax, ignoring the hierarchy entirely."""
        if level not in self.models:
            raise KeyError(f"No model for {level!r}. Trained levels: {list(self.models)}")
        X = self.transform(texts)
        model = self.models[level]
        preds = np.empty(X.shape[0], dtype=object)
        for start, end, X_batch in self._iter_batches(X):
            preds[start:end] = model.predict(X_batch)
        return preds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HierarchicalPathClassifier":
        import joblib

        return joblib.load(path)
