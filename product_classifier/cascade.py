"""Variable-depth prediction: emit the deepest prefix that is safe enough.

The fixed-depth question ("should we predict to L2 or L4?") forces one
answer onto every row. Most catalogues do not work that way: some
descriptions determine an L4 unambiguously, others barely pin down an L1.
A cascade decides per row, going as deep as the confidence supports and
truncating rather than guessing beyond it.

Two pieces:

    tune_thresholds()  pick a probability cut-off per depth, on held-out
                       data, so that each depth hits a target accuracy
    apply_cascade()    assign each row the deepest depth clearing its
                       cut-off, or 0 to abstain

Nothing here is imported by the rest of the package, and nothing here
modifies the model. It consumes the frame from
HierarchicalPathClassifier.predict_prefix_probabilities().

Typical use:

    prefix = result.classifier.predict_prefix_probabilities(test[text_col])
    actual = test[levels].astype(str).reset_index(drop=True)

    thresholds = tune_thresholds(prefix, actual, levels, target_accuracy=0.95)
    assigned   = apply_cascade(prefix, levels, thresholds)
    report     = cascade_report(assigned, actual, levels)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROB_FMT = "L{d} Prefix Probability"
MARGIN_FMT = "L{d} Margin"
PRED_FMT = "L{d} Predicted {level}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _depths(prefix_df: pd.DataFrame) -> List[int]:
    return sorted(
        int(c.split()[0][1:])
        for c in prefix_df.columns
        if c.endswith("Prefix Probability")
    )


def prefix_correct(
    prefix_df: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    depth: int,
) -> np.ndarray:
    """Boolean per row: is the predicted prefix at this depth fully right?"""
    cols = level_columns[:depth]
    pred = np.column_stack(
        [prefix_df[PRED_FMT.format(d=depth, level=c)].astype(str).values for c in cols]
    )
    truth = np.column_stack([actual[c].astype(str).values for c in cols])
    return (pred == truth).all(axis=1)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def tune_thresholds(
    prefix_df: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    target_accuracy: float = 0.95,
    min_coverage: float = 0.0,
    score: str = "probability",
) -> Dict[int, float]:
    """Lowest cut-off per depth that still meets the accuracy target.

    Thresholds are chosen empirically rather than by trusting the model's
    probabilities at face value: Naive Bayes is overconfident, so a
    nominal 0.95 does not mean 95% correct. Sweeping the observed scores
    and reading off realised accuracy sidesteps that entirely.

    Lowering the cut-off admits more rows at that depth and accuracy falls
    monotonically-ish, so the lowest qualifying cut-off is the one that
    maximises coverage subject to the accuracy floor.

    Returns {depth: threshold}. A depth that can never reach the target,
    at any cut-off, gets inf -- meaning it is never emitted.

    Tune this on data the model did not train on, and ideally not the same
    rows used to report final numbers: thresholds fitted and evaluated on
    one test set are optimistic.
    """
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0, 1].")
    fmt = PROB_FMT if score == "probability" else MARGIN_FMT

    thresholds = {}
    for d in _depths(prefix_df):
        s = prefix_df[fmt.format(d=d)].to_numpy(dtype=float)
        correct = prefix_correct(prefix_df, actual, level_columns, d)

        order = np.argsort(-s, kind="stable")
        hits = np.cumsum(correct[order])
        n_taken = np.arange(1, len(s) + 1)
        acc = hits / n_taken
        coverage = n_taken / len(s)

        ok = (acc >= target_accuracy) & (coverage >= min_coverage)
        thresholds[d] = float(s[order][np.flatnonzero(ok)[-1]]) if ok.any() else np.inf
    return thresholds


# ---------------------------------------------------------------------------
# Applying the cascade
# ---------------------------------------------------------------------------

def apply_cascade(
    prefix_df: pd.DataFrame,
    level_columns: List[str],
    thresholds: Dict[int, float],
    score: str = "probability",
) -> pd.DataFrame:
    """Assign each row the deepest depth whose score clears its threshold.

    Depths are tested deepest-first and the first pass wins. Rows clearing
    nothing get depth 0 and empty labels: an explicit abstention, routed
    to a human rather than guessed at.

    Returns the original level columns (empty string below the assigned
    depth), plus 'Assigned Depth' and 'Assigned Probability'.
    """
    fmt = PROB_FMT if score == "probability" else MARGIN_FMT
    n = len(prefix_df)
    depths = _depths(prefix_df)

    assigned = np.zeros(n, dtype=int)
    conf = np.zeros(n, dtype=float)
    for d in sorted(depths, reverse=True):
        s = prefix_df[fmt.format(d=d)].to_numpy(dtype=float)
        take = (assigned == 0) & (s >= thresholds.get(d, np.inf))
        assigned[take] = d
        conf[take] = s[take]

    out = {c: np.full(n, "", dtype=object) for c in level_columns}
    for d in depths:
        rows = assigned == d
        if not rows.any():
            continue
        for c in level_columns[:d]:
            out[c][rows] = prefix_df.loc[rows, PRED_FMT.format(d=d, level=c)].astype(str)

    frame = pd.DataFrame(out)
    frame["Assigned Depth"] = assigned
    frame["Assigned Probability"] = conf
    return frame


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def cascade_report(
    assigned: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
) -> pd.DataFrame:
    """Coverage and realised accuracy at each assigned depth.

    'accuracy' is measured only over rows assigned to that depth, against
    the prefix actually emitted -- so a row emitted at L2 is judged on L1
    and L2 only. That is the number a reviewer experiences.
    """
    rows = []
    n = len(assigned)
    depth_col = assigned["Assigned Depth"].to_numpy()

    for d in [0] + sorted(set(depth_col[depth_col > 0])):
        mask = depth_col == d
        n_d = int(mask.sum())
        if d == 0:
            acc = np.nan
        else:
            cols = level_columns[:d]
            pred = assigned.loc[mask, cols].astype(str).values
            truth = actual.loc[mask, cols].astype(str).values
            acc = float((pred == truth).all(axis=1).mean()) if n_d else np.nan
        rows.append(
            {
                "assigned_depth": d,
                "n_items": n_d,
                "share_of_items": n_d / n if n else np.nan,
                "accuracy": acc,
            }
        )

    report = pd.DataFrame(rows)
    emitted = report[report["assigned_depth"] > 0]
    total_emitted = emitted["n_items"].sum()
    report.attrs["mean_depth_all_rows"] = (
        float((emitted["assigned_depth"] * emitted["n_items"]).sum() / n) if n else np.nan
    )
    report.attrs["mean_depth_emitted"] = (
        float((emitted["assigned_depth"] * emitted["n_items"]).sum() / total_emitted)
        if total_emitted
        else np.nan
    )
    report.attrs["abstain_rate"] = (
        float(report.loc[report["assigned_depth"] == 0, "n_items"].iloc[0] / n)
        if n
        else np.nan
    )
    return report


# ---------------------------------------------------------------------------
# Per-item output
# ---------------------------------------------------------------------------

def per_item_table(
    assigned: pd.DataFrame,
    source_df: pd.DataFrame,
    level_columns: List[str],
    text_col: str,
    id_col: Optional[str] = None,
    prefix_df: Optional[pd.DataFrame] = None,
    actual: Optional[pd.DataFrame] = None,
    quality: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per product: how deep to trust the model, and the answer.

    This is the deliverable. Everything else in the module exists to
    decide what goes in the 'Assigned Depth' column.

    'Auto Accept' is the yes/no a reviewer cares about: the model went all
    the way, so nobody needs to look. Below full depth the level columns
    are filled only as far as the model committed, and the blanks are the
    work left over.

    Pass actual and prefix_df on labelled data to get a 'Prefix Correct'
    column for auditing. On new products, leave both out -- there is
    nothing to compare against, which is the normal production case.
    """
    full = len(level_columns)
    keep = ([id_col] if id_col else []) + [text_col]

    parts = [source_df.reset_index(drop=True)[keep], assigned.reset_index(drop=True)]
    if quality is not None:
        parts.append(
            quality.reset_index(drop=True)[["n_tokens", "n_known", "oov_rate"]]
        )
    out = pd.concat(parts, axis=1)

    out["Auto Accept"] = out["Assigned Depth"] == full
    out["Levels Left To Do"] = full - out["Assigned Depth"]
    out["Outcome"] = out["Assigned Depth"].map(
        lambda d: "auto-accept" if d == full
        else ("hand over entirely" if d == 0 else "partial - human completes")
    )

    if prefix_df is not None and actual is not None:
        correct = np.zeros(len(out), dtype=bool)
        for d in _depths(prefix_df):
            rows = (out["Assigned Depth"] == d).to_numpy()
            if rows.any():
                correct[rows] = prefix_correct(prefix_df, actual, level_columns, d)[rows]
        # A row that was handed over has no prediction to be right or wrong.
        out["Prefix Correct"] = np.where(
            out["Assigned Depth"].to_numpy() > 0, correct, np.nan
        )
    return out


def outcome_summary(per_item: pd.DataFrame) -> pd.DataFrame:
    """Counts and shares by outcome, plus accuracy where it is known."""
    rows = []
    n = len(per_item)
    for name, grp in per_item.groupby("Outcome"):
        row = {"outcome": name, "n_items": len(grp), "share": len(grp) / n}
        if "Prefix Correct" in per_item.columns:
            vals = grp["Prefix Correct"].dropna()
            row["accuracy"] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_items", ascending=False)
    out.attrs["mean_depth"] = float(per_item["Assigned Depth"].mean())
    return out.reset_index(drop=True)
