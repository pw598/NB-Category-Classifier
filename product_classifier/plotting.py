"""Charts. Every function returns the Figure so the caller decides whether
to show it, save it, or log it to MLflow.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

BAR_COLOR = "steelblue"
LINE_COLOR = "firebrick"


def _accept_boundary(bucket_summary: pd.DataFrame,
                     threshold: Optional[float],
                     target_accuracy: Optional[float]) -> Optional[tuple]:
    """Where to draw the accept boundary on a categorical bucket axis.

    Returns (x_position, label) or None if nothing qualifies.

    The x axis is a row of bars, one per bucket, so a threshold cannot be
    placed at its true value inside a bucket. It is rounded *up* to the
    bucket edge above it, which is the conservative direction: a bucket
    only partly above the threshold is excluded rather than counted. The
    line sits at position j - 0.5, meaning every bar to its right is
    accepted.

    With an explicit threshold, j is the first bucket lying entirely above
    it. Without one, j is the lowest bucket whose cumulative accuracy from
    the top still meets target_accuracy -- the same question answered from
    the table instead.
    """
    if threshold is not None and "bucket_left" in bucket_summary.columns:
        qualifying = bucket_summary.index[
            bucket_summary["bucket_left"].to_numpy() >= threshold
        ]
        if not len(qualifying):
            return None
        j = int(qualifying[0])
        return j - 0.5, f"threshold {threshold:.4g}"

    if target_accuracy is not None and "cumulative_accuracy" in bucket_summary.columns:
        ok = bucket_summary.index[
            bucket_summary["cumulative_accuracy"].to_numpy() >= target_accuracy
        ]
        if not len(ok):
            return None
        j = int(ok[0])
        cov = bucket_summary["cumulative_coverage"].iloc[j]
        return j - 0.5, f"{target_accuracy:.0%} at {cov:.1%} coverage"
    return None


def _draw_accuracy_by_bucket(ax1, bucket_summary: pd.DataFrame, annotate: bool = True,
                             target_accuracy: Optional[float] = None,
                             threshold: Optional[float] = None):
    """Volume bars against an accuracy line, sharing the confidence axis."""
    x = bucket_summary["Probability Bucket"].astype(str)
    ax1.bar(x, bucket_summary["n_items"], color=BAR_COLOR, alpha=0.7)
    ax1.set_xlabel("Probability bucket")
    ax1.set_ylabel("Number of items", color=BAR_COLOR)
    ax1.tick_params(axis="y", labelcolor=BAR_COLOR)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, bucket_summary["accuracy"], color=LINE_COLOR, marker="o", linewidth=2)
    ax2.set_ylabel("Path accuracy", color=LINE_COLOR)
    ax2.tick_params(axis="y", labelcolor=LINE_COLOR)
    ax2.set_ylim(0, 1.05)

    if annotate:
        for xi, yi in zip(x, bucket_summary["accuracy"]):
            if pd.notna(yi):
                ax2.annotate(
                    f"{yi:.1%}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=LINE_COLOR,
                )
        total = bucket_summary["n_items"].sum()
        if total:
            for xi, n in zip(x, bucket_summary["n_items"]):
                ax1.annotate(
                    f"{n / total:.1%}", (xi, 0), textcoords="offset points",
                    xytext=(0, 3), ha="center", va="bottom", fontsize=9, color="black",
                )

    if target_accuracy is not None:
        ax2.axhline(target_accuracy, linestyle="--", color="gray", linewidth=1)

    boundary = _accept_boundary(bucket_summary, threshold, target_accuracy)
    if boundary is not None:
        pos, label = boundary
        ax1.axvline(pos, linestyle="--", color="darkgreen", linewidth=1.2)
        ax1.annotate(
            label, (pos, ax1.get_ylim()[1]), textcoords="offset points",
            xytext=(3, -10), fontsize=8, color="darkgreen",
            rotation=90, va="top",
        )
    return ax2


def plot_accuracy_by_bucket(
    bucket_summary: pd.DataFrame,
    title: str = "Path accuracy and item count by probability bucket",
    figsize: tuple = (10, 6),
    target_accuracy: Optional[float] = None,
    threshold: Optional[float] = None,
) -> plt.Figure:
    """Volume bars against an accuracy line, sharing the confidence axis.

    target_accuracy draws a dashed horizontal line on the accuracy axis.
    threshold draws a dashed vertical accept boundary, rounded up to a
    bucket edge; omit it and the boundary is taken from the point where
    cumulative accuracy first meets target_accuracy.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    _draw_accuracy_by_bucket(ax1, bucket_summary, target_accuracy=target_accuracy,
                             threshold=threshold)
    ax1.set_title(title)
    fig.tight_layout()
    return fig


def _draw_coverage_curve(ax, bucket_summary: pd.DataFrame,
                         target_accuracy: Optional[float] = 0.95,
                         annotate: bool = True):
    ax.plot(
        bucket_summary["cumulative_coverage"],
        bucket_summary["cumulative_accuracy"],
        marker="o", color=LINE_COLOR,
    )
    if annotate:
        for _, row in bucket_summary.iterrows():
            if pd.notna(row["cumulative_coverage"]):
                ax.annotate(
                    str(row["Probability Bucket"]),
                    (row["cumulative_coverage"], row["cumulative_accuracy"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8,
                )
    if target_accuracy is not None:
        ax.axhline(target_accuracy, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Coverage (share of items auto-classified)")
    ax.set_ylabel("Accuracy on accepted items")
    ax.set_ylim(0, 1.05)


def plot_coverage_curve(
    bucket_summary: pd.DataFrame,
    target_accuracy: Optional[float] = 0.95,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Accuracy against coverage as the auto-accept threshold moves.

    Read it as: to hit `target_accuracy`, this is the share of items you
    can classify without review.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _draw_coverage_curve(ax, bucket_summary, target_accuracy)
    ax.set_title("Accuracy vs. coverage (cumulative)")
    fig.tight_layout()
    return fig


def _grid(n, ncols, figsize, per_panel=(7.5, 4.6)):
    import numpy as _np
    nrows = int(_np.ceil(n / ncols))
    figsize = figsize or (per_panel[0] * ncols, per_panel[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    return fig, axes, nrows


def _target_for(target, depth):
    """target may be one number for every level, or {depth: number}."""
    if isinstance(target, dict):
        return target.get(depth)
    return target


def _coverage_at_threshold(bucket_summary: pd.DataFrame,
                           threshold: Optional[float]) -> Optional[float]:
    """Coverage delivered by a threshold, rounded up to a bucket edge.

    Same rounding as the accept boundary on the bar chart, so the two
    charts mark the same decision.
    """
    if threshold is None or "bucket_left" not in bucket_summary.columns:
        return None
    q = bucket_summary.index[bucket_summary["bucket_left"].to_numpy() >= threshold]
    if not len(q):
        return None
    return float(bucket_summary["cumulative_coverage"].iloc[int(q[0])])


def _panel_title(depth, level_columns):
    if level_columns and depth <= len(level_columns):
        return f"Correct through {level_columns[depth - 1]}"
    return f"Depth {depth}"


def plot_accuracy_by_bucket_grid(
    summaries: dict,
    level_columns: Optional[list] = None,
    ncols: int = 2,
    figsize: Optional[tuple] = None,
    annotate: bool = False,
    target_accuracy=0.95,
    thresholds: Optional[dict] = None,
) -> plt.Figure:
    """One accuracy-and-volume chart per level.

    target_accuracy may be a single number or {depth: number}; a cost
    model implies a different target at each level.

    summaries is {depth: bucket_summary}, as returned by
    repeated_holdout.level_bucket_summary().

    target_accuracy draws a dashed horizontal line on each accuracy axis.

    thresholds is {depth: threshold} -- pass the output of
    tune_thresholds() to mark where each level's accept boundary falls.
    Left as None, the boundary is read off each panel's own cumulative
    accuracy instead, which needs no extra input and answers the same
    question from the table.

    Either way the vertical line is rounded up to a bucket edge, since the
    x axis is a row of bars rather than a continuous scale. Everything to
    the right of the line is accepted.

    Annotations are off by default here: the per-bucket labels that fit a
    full-size chart collide once four of them share a page.
    """
    depths = sorted(summaries)
    fig, axes, nrows = _grid(len(depths), ncols, figsize)
    for i, d in enumerate(depths):
        ax = axes[i // ncols][i % ncols]
        _draw_accuracy_by_bucket(
            ax, summaries[d], annotate=annotate,
            target_accuracy=_target_for(target_accuracy, d),
            threshold=(thresholds or {}).get(d),
        )
        t = _target_for(target_accuracy, d)
        title = _panel_title(d, level_columns)
        if t is not None:
            title += f"  (target {t:.1%})"
        ax.set_title(title, fontsize=10)
    for i in range(len(depths), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle("Accuracy and item count by probability bucket, per level", y=1.0)
    fig.tight_layout()
    return fig


def plot_coverage_curve_grid(
    summaries: dict,
    target_accuracy=0.95,
    level_columns: Optional[list] = None,
    ncols: int = 2,
    figsize: Optional[tuple] = None,
    annotate: bool = False,
    thresholds: Optional[dict] = None,
) -> plt.Figure:
    """One coverage curve per level.

    Each panel answers, for that depth: to hit the target, what share of
    products can be auto-classified? The dashed horizontal is the target,
    so where the curve crosses it is the answer.

    target_accuracy may be a single number or {depth: number}. Per-level
    targets are what a cost model implies -- the accuracy each level has
    to clear differs, because stepping one level deeper saves a different
    amount of staff time at each depth.

    thresholds marks the coverage each level's threshold actually
    delivers, using the same rounding as the bar chart so both charts
    describe the same decision.
    """
    depths = sorted(summaries)
    fig, axes, nrows = _grid(len(depths), ncols, figsize, per_panel=(6.2, 4.2))
    for i, d in enumerate(depths):
        ax = axes[i // ncols][i % ncols]
        target = _target_for(target_accuracy, d)
        _draw_coverage_curve(ax, summaries[d], target, annotate=annotate)
        cov = _coverage_at_threshold(summaries[d], (thresholds or {}).get(d))
        if cov is not None:
            ax.axvline(cov, linestyle="--", color="darkgreen", linewidth=1.2)
            ax.annotate(f"{cov:.1%} coverage", (cov, 0.03),
                        textcoords="offset points", xytext=(4, 0),
                        fontsize=8, color="darkgreen")
        title = _panel_title(d, level_columns)
        if target is not None:
            title += f"  (target {target:.1%})"
        ax.set_title(title, fontsize=10)
    for i in range(len(depths), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle("Accuracy vs. coverage (cumulative), per level", y=1.0)
    fig.tight_layout()
    return fig


def plot_per_level_accuracy(
    per_level: dict, figsize: tuple = (8, 4.5)
) -> plt.Figure:
    """Where in the hierarchy accuracy falls off."""
    fig, ax = plt.subplots(figsize=figsize)
    levels = list(per_level)
    values = [per_level[k] for k in levels]
    ax.bar(levels, values, color=BAR_COLOR)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.1%}", (i, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by hierarchy level")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_top_confusions(
    confusion: pd.DataFrame, n: int = 15, figsize: tuple = (9, 6)
) -> plt.Figure:
    """Horizontal bars for the most common actual -> predicted mix-ups."""
    cols = list(confusion.columns)
    actual_col, pred_col = cols[0], cols[1]
    top = confusion.head(n).iloc[::-1]
    labels = top[actual_col].astype(str) + "  ->  " + top[pred_col].astype(str)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, top["n"], color=BAR_COLOR)
    ax.set_xlabel("Misclassified items")
    ax.set_title("Most frequent confusions")
    fig.tight_layout()
    return fig
