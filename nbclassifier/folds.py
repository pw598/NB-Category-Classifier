"""Fold assignment.

id_hash_folds is byte-for-byte the method used by NB-Category-Classifier and
the neural-network repo, so out-of-fold predictions from both models line up
for the ensemble. Keep N_FOLDS, FOLD_SEED and FOLD_KEY_COL identical across
the two projects.
"""

from __future__ import annotations

import hashlib
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

_BAD_KEYS = {"", "nan", "none", "<na>"}


def id_hash_folds(ids, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold of each row determined only by its ID: sha1(f"{seed}|{id}")."""
    keys = pd.Series(ids, dtype=object).astype(str).str.strip()
    bad = keys.str.lower().isin(_BAD_KEYS)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) have a blank/null fold key; fix or drop them first")
    out = np.empty(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        digest = hashlib.sha1(f"{seed}|{key}".encode("utf-8")).digest()
        out[i] = int.from_bytes(digest[:8], "big") % n_folds
    return out


def id_hash_sample(ids, frac: float, seed: int = 0) -> np.ndarray:
    """Boolean mask keeping about `frac` of the rows, decided by ID alone.

    sha1(f"sample|{seed}|{id}") mapped to [0, 1). Like the folds it does not
    depend on which other rows are present, so the NB and NN repos sample the
    same items for the same SAMPLE_FRAC and SAMPLE_SEED.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    keys = pd.Series(ids, dtype=object).astype(str).str.strip()
    out = np.empty(len(keys), dtype=bool)
    for i, key in enumerate(keys):
        digest = hashlib.sha1(f"sample|{seed}|{key}".encode("utf-8")).digest()
        out[i] = int.from_bytes(digest[:8], "big") / 2.0 ** 64 < frac
    return out


def stratified_folds(labels, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """StratifiedKFold on the given labels (deepest level). Classes with fewer
    members than n_folds are spread round-robin instead of raising."""
    labels = pd.Series(labels).astype(str).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = np.empty(len(labels), dtype=np.int64)
    for _, idx in labels.groupby(labels).indices.items():
        idx = rng.permutation(idx)
        start = rng.integers(n_folds)
        out[idx] = (start + np.arange(len(idx))) % n_folds
    return out


def holdout_splits(n_rows: int, n_repeats: int, test_size: float, seed: int) -> Iterator[Tuple[int, np.ndarray]]:
    """(repeat, boolean test mask) for repeated random holdout."""
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        mask = np.zeros(n_rows, dtype=bool)
        mask[rng.choice(n_rows, size=int(round(test_size * n_rows)), replace=False)] = True
        yield r, mask
