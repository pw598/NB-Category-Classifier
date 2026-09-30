"""Scoring, kept separate from the model.

The classifier's job is to turn text into predictions. Judging those
predictions is a different concern with a different rate of change (new
metrics get added often; the model rarely changes), so it lives here and
takes plain DataFrames rather than a fitted estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .model import PROBABILITY_COL, HierarchicalPathClassifier

ACTUAL_PREFIX = "Actual "
PREDICTED_PREFIX = "Predicted "


@dataclass
class EvaluationResult:
    """Everything one scoring pass produces."""

    metrics: Dict[str, float] = field(default_factory=dict)
    per_level_accuracy: Dict[str, float] = field(default_factory=dict)
    comparison: Optional[pd.DataFrame] = None
    bucket_summary: Optional[pd.DataFrame] = None
    confusion: Optional[pd.DataFrame] = None

    def summary_text(self) -> str:
        lines = [f"{k}: {v:.2%}" for k, v in self.metrics.items()]
        lines += [f"  {k}: {v:.2%}" for k, v in self.per_level_accuracy.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def path_accuracy(
    predicted: pd.DataFrame, actual: pd.DataFrame, level_columns: List[str]
) -> float:
    """Fraction of rows where every level matches."""
    match = (
        predicted[level_columns].astype(str).values
        == actual[level_columns].astype(str).values
    ).all(axis=1)
    return float(match.mean())


def per_level_accuracy(
    predicted: pd.DataFrame, actual: pd.DataFrame, level_columns: List[str]
) -> Dict[str, float]:
    return {
        level: float(
            (
                predicted[level].astype(str).values == actual[level].astype(str).values
            ).mean()
        )
        for level in level_columns
    }


def prefix_accuracy(
    predicted: pd.DataFrame, actual: pd.DataFrame, level_columns: List[str]
) -> Dict[str, float]:
    """Accuracy of the path truncated at each depth.

    'Correct through L2' is often the number a business cares about even
    when L4 is wrong, so report the whole ladder rather than only the
    all-or-nothing path metric.
    """
    out = {}
    for depth in range(1, len(level_columns) + 1):
        cols = level_columns[:depth]
        out[f"correct_through_{cols[-1]}"] = path_accuracy(predicted, actual, cols)
    return out


# ---------------------------------------------------------------------------
# Comparison frame and calibration buckets
# ---------------------------------------------------------------------------

def build_comparison(
    source_df: pd.DataFrame,
    predicted: pd.DataFrame,
    level_columns: List[str],
    text_col: str,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Side-by-side actual vs. predicted, one row per test item."""
    source_df = source_df.reset_index(drop=True)
    predicted = predicted.reset_index(drop=True)

    keep = ([id_col] if id_col else []) + [text_col]
    actual = source_df[keep + level_columns].rename(
        columns={level: f"{ACTUAL_PREFIX}{level}" for level in level_columns}
    )
    pred_cols = [c for c in predicted.columns if c in level_columns] + [PROBABILITY_COL]
    pred = predicted[pred_cols].rename(
        columns={level: f"{PREDICTED_PREFIX}{level}" for level in level_columns}
    )

    comparison = pd.concat([actual, pred], axis=1)
    comparison["Path Correct"] = (
        comparison[[f"{PREDICTED_PREFIX}{c}" for c in level_columns]].astype(str).values
        == comparison[[f"{ACTUAL_PREFIX}{c}" for c in level_columns]].astype(str).values
    ).all(axis=1)
    return comparison


def probability_buckets(
    comparison: pd.DataFrame, bucket_width: float = 0.1
) -> pd.DataFrame:
    """Volume and accuracy by confidence band.

    This is the table that tells you where to set an auto-accept threshold:
    if the 90-100% band is 95% accurate and holds 40% of items, those 40%
    need no human review.
    """
    edges = np.arange(0, 1.0 + bucket_width / 2, bucket_width)
    labels = [
        f"{int(round(edges[i] * 100))}-{int(round(edges[i + 1] * 100))}%"
        for i in range(len(edges) - 1)
    ]
    out = comparison.copy()
    out["Probability Bucket"] = pd.cut(
        out[PROBABILITY_COL], bins=edges, labels=labels, include_lowest=True
    )
    summary = (
        out.groupby("Probability Bucket", observed=False)
        .agg(n_items=("Path Correct", "size"), accuracy=("Path Correct", "mean"))
        .reset_index()
    )
    total = summary["n_items"].sum()
    summary["share_of_items"] = summary["n_items"] / total if total else np.nan
    # Cumulative view from the top down: "if I auto-accept everything above
    # this bucket, what coverage and accuracy do I get?"
    rev = summary.iloc[::-1]
    cum_n = rev["n_items"].cumsum()
    cum_correct = (rev["n_items"] * rev["accuracy"].fillna(0)).cumsum()
    summary["cumulative_coverage"] = (cum_n / total).iloc[::-1].values if total else np.nan
    summary["cumulative_accuracy"] = (cum_correct / cum_n).iloc[::-1].values
    return summary


def top_confusions(
    comparison: pd.DataFrame, level: str, n: int = 20
) -> pd.DataFrame:
    """The most frequent actual -> predicted mistakes at one level."""
    a, p = f"{ACTUAL_PREFIX}{level}", f"{PREDICTED_PREFIX}{level}"
    wrong = comparison[comparison[a].astype(str) != comparison[p].astype(str)]
    return (
        wrong.groupby([a, p]).size().reset_index(name="n").sort_values("n", ascending=False).head(n)
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate(
    clf: HierarchicalPathClassifier,
    test_df: pd.DataFrame,
    text_col: str,
    id_col: Optional[str] = None,
    bucket_width: float = 0.1,
    compare_against_per_level: bool = True,
    confusion_level: Optional[str] = None,
) -> EvaluationResult:
    """Score a fitted classifier on a held-out frame."""
    levels = clf.level_columns
    actual = test_df[levels].astype(str).reset_index(drop=True)

    predicted = clf.predict(test_df[text_col])
    comparison = build_comparison(test_df, predicted, levels, text_col, id_col)

    metrics = {"path_accuracy": path_accuracy(predicted, actual, levels)}
    metrics.update(prefix_accuracy(predicted, actual, levels))

    if (
        compare_against_per_level
        and clf.config.prediction_mode in ("joint_path", "conditional_path")
        and clf.models
    ):
        flat = clf.predict(test_df[text_col], mode="per_level")
        metrics["per_level_mode_path_accuracy"] = path_accuracy(flat, actual, levels)
        deepest = levels[-1]
        metrics[f"per_level_mode_{deepest}_accuracy"] = float(
            (flat[deepest].astype(str).values == actual[deepest].values).mean()
        )

    return EvaluationResult(
        metrics=metrics,
        per_level_accuracy=per_level_accuracy(predicted, actual, levels),
        comparison=comparison,
        bucket_summary=probability_buckets(comparison, bucket_width),
        confusion=top_confusions(comparison, confusion_level or levels[-1]),
    )
