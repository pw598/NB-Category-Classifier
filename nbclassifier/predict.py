"""The prediction table shared by out-of-fold evaluation and batch prediction.

Both write the same columns so the ensemble can read either:

    <id columns> [fold] [Actual <level> ...]
    Predicted <level> ...                 e.g. 'Predicted Level 4 ID'
    [Predicted <name level> ...]          e.g. 'Predicted Level 4 Name' (when a lookup is available)
    L1..L4 Confidence                     raw summed path score of the predicted prefix
    L1..L4 Margin                         confidence minus best competing prefix at that depth
    n_tokens, n_known, oov_rate           input quality against the model vocabulary
    [<raw text col>, <cleaned text col>]

Files are comma-separated CSV (what the ensemble reads).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def attach_names(pred: pd.DataFrame, lookup: Optional[pd.DataFrame],
                 level_columns: Sequence[str], name_columns: Optional[Sequence[str]]) -> pd.DataFrame:
    if lookup is None or not name_columns:
        return pred
    out = pred.copy()
    insert_at = list(out.columns).index(f"Predicted {level_columns[-1]}") + 1
    for lvl, name in zip(level_columns, name_columns):
        if lvl in lookup.columns and name in lookup.columns:
            mapping = lookup[[lvl, name]].dropna().drop_duplicates(lvl).set_index(lvl)[name]
            out.insert(insert_at, f"Predicted {name}", out[f"Predicted {lvl}"].map(mapping))
            insert_at += 1
    return out


def prediction_table(
    clf,                              # any fitted classifier with predict_frame() and input_quality()
    df: pd.DataFrame,
    cleaned_col: str,
    id_cols: Sequence[str] = (),
    actual_cols: Sequence[str] = (),
    extra: Optional[Dict[str, np.ndarray]] = None,
    text_cols: Sequence[str] = (),
    X=None,
    batch_size: int = 2000,
) -> pd.DataFrame:
    """Predictions for every row of df (row order preserved)."""
    pred = clf.predict_frame(df[cleaned_col], X=X, batch_size=batch_size)
    quality = clf.input_quality(df[cleaned_col])
    parts = [df[[c for c in id_cols if c in df.columns]].reset_index(drop=True)]
    if extra:
        parts.append(pd.DataFrame(extra))
    if actual_cols:
        parts.append(df[list(actual_cols)].reset_index(drop=True)
                     .rename(columns={c: f"Actual {c}" for c in actual_cols}))
    parts += [pred, quality]
    if text_cols:
        parts.append(df[[c for c in text_cols if c in df.columns]].reset_index(drop=True))
    return pd.concat(parts, axis=1)


def accuracy_by_level(pred: pd.DataFrame, level_columns: Sequence[str]) -> Dict[str, float]:
    """Path-prefix accuracy: level d is correct when levels 1..d all match."""
    out, ok = {}, None
    for d, col in enumerate(level_columns, start=1):
        hit = (pred[f"Predicted {col}"].astype(str).to_numpy()
               == pred[f"Actual {col}"].astype(str).to_numpy())
        ok = hit if ok is None else (ok & hit)
        out[f"L{d} Accuracy"] = float(ok.mean()) if len(ok) else float("nan")
    return out


def correct_flags(pred: pd.DataFrame, level_columns: Sequence[str]) -> pd.DataFrame:
    flags, ok = {}, None
    for d, col in enumerate(level_columns, start=1):
        hit = (pred[f"Predicted {col}"].astype(str).to_numpy()
               == pred[f"Actual {col}"].astype(str).to_numpy())
        ok = hit if ok is None else (ok & hit)
        flags[f"L{d} Correct"] = ok
    return pd.DataFrame(flags, index=pred.index)
