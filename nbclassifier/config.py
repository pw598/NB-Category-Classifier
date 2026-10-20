"""Hyperparameters and evaluation settings.

There is deliberately no hidden default here that the notebooks rely on:
every notebook spells out every hyperparameter. The defaults below exist only
so the classes can be constructed in tests.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

PREDICTION_MODES = ("independent", "conditional", "flattened")

# How the vectorizer splits cleaned text into tokens.
#   whitespace: every space-separated word is a token, exactly as the cleaning method
#               defines words - keeps <SIZE>IN, <COLOR> and single characters intact
#   sklearn:    sklearn's default, runs of 2+ word characters (<SIZE>IN -> 'size', 'in')
TOKEN_PATTERNS = {"whitespace": r"(?u)\S+", "sklearn": r"(?u)\b\w\w+\b"}
VECTORIZER_KINDS = ("tfidf", "count")

# Parameters that only matter for one vectorizer kind. Used to de-duplicate
# grid candidates that would otherwise be identical models.
TFIDF_ONLY = ("sublinear_tf", "norm", "use_idf")


@dataclass
class HyperParams:
    """Every tunable setting of the model, flat, with notebook-friendly names."""

    # --- vectorizer -------------------------------------------------------
    vectorizer_kind: str = "tfidf"          # 'tfidf' or 'count'
    max_features: Optional[int] = None      # None = keep the whole vocabulary
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: Any = 2                         # int = documents, float = proportion
    max_df: Any = 0.95
    stop_words: Optional[str] = "english"
    sublinear_tf: bool = True               # tfidf only
    binary: bool = False
    norm: Optional[str] = "l2"              # tfidf only: 'l1', 'l2' or None
    use_idf: bool = True                    # tfidf only
    ordered_bigrams: bool = True            # False: 'A B' and 'B A' are the same bigram
    token_pattern: str = "whitespace"       # 'whitespace' | 'sklearn' | a regex (see TOKEN_PATTERNS)
    # --- naive Bayes ------------------------------------------------------
    alpha: float = 0.001
    fit_prior: bool = True
    prediction_mode: str = "conditional"    # 'independent' | 'conditional' | 'flattened'
    # --- data -------------------------------------------------------------
    rare_path_threshold: int = 5            # drop paths seen fewer times than this

    def __post_init__(self):
        if isinstance(self.ngram_range, list):
            self.ngram_range = tuple(self.ngram_range)
        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.vectorizer_kind not in VECTORIZER_KINDS:
            raise ValueError(f"vectorizer_kind must be one of {VECTORIZER_KINDS}")
        if self.prediction_mode not in PREDICTION_MODES:
            raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
        if not (isinstance(self.alpha, (int, float)) and self.alpha > 0):
            raise ValueError("alpha must be a positive number")
        lo, hi = self.ngram_range
        if not (1 <= lo <= hi):
            raise ValueError(f"invalid ngram_range {self.ngram_range}")
        if not self.ordered_bigrams and not (lo <= 2 <= hi):
            raise ValueError("ordered_bigrams=False needs 2 inside ngram_range")
        if int(self.rare_path_threshold) < 1:
            raise ValueError("rare_path_threshold must be >= 1")
        if isinstance(self.min_df, float) and self.min_df >= 1.0:
            raise ValueError("a float min_df is a proportion and must be < 1.0")
        if isinstance(self.max_df, int) and not isinstance(self.max_df, bool) and self.max_df <= 1:
            raise ValueError("an int max_df is a document count; use a float (e.g. 1.0) for a proportion")

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ngram_range"] = list(self.ngram_range)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HyperParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def effective(self) -> Dict[str, Any]:
        """Settings that actually change the model (tfidf-only ones blanked for 'count')."""
        d = self.to_dict()
        if self.vectorizer_kind == "count":
            for k in TFIDF_ONLY:
                d[k] = None
        return d

    def vectorizer_key(self) -> str:
        """Identifies everything that determines the vectorized data.

        Grid candidates sharing this key share the vectorizer, the rare-path
        filter and the per-level counts, so they are fitted once per fold.
        """
        d = self.effective()
        keep = [
            "vectorizer_kind", "max_features", "ngram_range", "min_df", "max_df",
            "stop_words", "sublinear_tf", "binary", "norm", "use_idf",
            "ordered_bigrams", "token_pattern", "rare_path_threshold",
        ]
        return json.dumps({k: d[k] for k in keep}, sort_keys=True)

    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.to_dict().items())


# ---------------------------------------------------------------------------
# Grid handling
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> List[Any]:
    # A tuple is a single value (e.g. ngram_range=(1, 2)); only lists are grids.
    return list(value) if isinstance(value, list) else [value]


def expand_grid(spec: Dict[str, Any], grid_search: bool) -> List[HyperParams]:
    """Turn the notebook's HYPERPARAMETERS dict into candidate HyperParams.

    With grid_search=False every entry must be a scalar or a one-element list.
    Candidates that would produce identical models (e.g. different 'norm'
    values with a count vectorizer) are de-duplicated.
    """
    known = {f.name for f in fields(HyperParams)}
    unknown = set(spec) - known
    if unknown:
        raise KeyError(f"Unknown hyperparameter(s): {sorted(unknown)}. Valid: {sorted(known)}")
    missing = known - set(spec)
    if missing:
        raise KeyError(
            f"Hyperparameter(s) not specified: {sorted(missing)}. "
            "Every hyperparameter must be set explicitly in the notebook."
        )

    ngr = spec.get("ngram_range")
    if isinstance(ngr, list) and ngr and all(isinstance(x, int) for x in ngr):
        raise ValueError(
            f"ngram_range={ngr} looks like a single range written as a list. "
            "Write it as a tuple, e.g. (1, 2), or as a list of tuples to search, e.g. [(1, 1), (1, 2)]."
        )
    lists = {k: _as_list(v) for k, v in spec.items()}
    if not grid_search:
        multi = {k: v for k, v in lists.items() if len(v) != 1}
        if multi:
            raise ValueError(
                "GRID_SEARCH = False, so every hyperparameter must be a scalar or a "
                f"single-element list. These have several values: {sorted(multi)}"
            )

    keys = list(lists)
    seen, out = set(), []
    for combo in itertools.product(*(lists[k] for k in keys)):
        hp = HyperParams(**dict(zip(keys, combo)))
        sig = json.dumps(hp.effective(), sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            out.append(hp)
    return out


@dataclass
class CVSettings:
    """How the evaluation splits the data."""

    cv_type: str = "oof"            # 'oof' (k-fold, out-of-fold) | 'repeated_holdout'
    n_folds: int = 5                # oof
    fold_method: str = "id_hash"    # oof: 'id_hash' (matches the NN repo) | 'stratified'
    fold_key_col: str = "ID"        # oof id_hash: column hashed to assign folds
    fold_seed: int = 0              # oof: hash seed / shuffle seed
    n_repeats: int = 5              # repeated_holdout
    test_size: float = 0.2          # repeated_holdout
    holdout_seed: int = 42          # repeated_holdout

    def __post_init__(self):
        self.cv_type = self.cv_type.lower()
        if self.cv_type not in ("oof", "repeated_holdout"):
            raise ValueError("cv_type must be 'oof' or 'repeated_holdout'")
        if self.fold_method not in ("id_hash", "stratified"):
            raise ValueError("fold_method must be 'id_hash' or 'stratified'")
        if self.cv_type == "oof" and self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if self.cv_type == "repeated_holdout" and not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def relevant(self) -> Dict[str, Any]:
        """Only the settings that affect this cv_type (used to validate a resume)."""
        if self.cv_type == "oof":
            keys = ["cv_type", "n_folds", "fold_method", "fold_seed"]
            if self.fold_method == "id_hash":
                keys.append("fold_key_col")
        else:
            keys = ["cv_type", "n_repeats", "test_size", "holdout_seed"]
        return {k: getattr(self, k) for k in keys}


def params_table(candidates: Sequence[HyperParams]):
    """Candidates as a DataFrame, one row each (for display)."""
    import pandas as pd

    rows = []
    for i, hp in enumerate(candidates):
        d = hp.to_dict()
        d["ngram_range"] = tuple(d["ngram_range"])
        rows.append({"candidate": i, **d})
    return pd.DataFrame(rows).set_index("candidate")
