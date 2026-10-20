"""Accuracy by confidence bucket, and the plots built on it.

Everything here works from a prediction table (see predict.py) that holds
'Actual <level>' columns, so it serves notebook 02 (out-of-fold
predictions) and notebook 04 (labelled batch predictions) alike. Bucket
width is an argument everywhere, so it can be changed and re-run after
the expensive cross-validation is finished.

Confidences are raw model scores; nothing is calibrated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .predict import accuracy_by_level, correct_flags


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _bucket_index(conf: np.ndarray, width: float) -> np.ndarray:
    nb = int(np.ceil(1.0 / width - 1e-9))
    idx = np.floor(np.clip(conf, 0.0, 1.0) / width + 1e-12).astype(int)
    return np.clip(idx, 0, nb - 1)


def _bucket_labels(width: float) -> List[str]:
    nb = int(np.ceil(1.0 / width - 1e-9))
    return [f"{i * width:.2f}-{min((i + 1) * width, 1.0):.2f}" for i in range(nb)]


def bucket_table(pred: pd.DataFrame, level_columns: Sequence[str], depth: int,
                 width: float = 0.1) -> pd.DataFrame:
    """Items, share and accuracy per confidence bucket at one depth (1-based).

    cumulative_coverage / cumulative_accuracy read from the top bucket down:
    the share of items at or above the bucket's lower edge, and their accuracy.
    """
    conf = pred[f"L{depth} Confidence"].to_numpy(dtype=float)
    ok = correct_flags(pred, level_columns)[f"L{depth} Correct"].to_numpy()
    idx = _bucket_index(conf, width)
    labels = _bucket_labels(width)
    nb = len(labels)
    n = np.bincount(idx, minlength=nb)
    hits = np.bincount(idx, weights=ok.astype(float), minlength=nb)
    total = max(len(conf), 1)
    cum_n = np.cumsum(n[::-1])[::-1]
    cum_hits = np.cumsum(hits[::-1])[::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        tbl = pd.DataFrame({
            "level": f"L{depth}",
            "bucket": labels,
            "lower": np.arange(nb) * width,
            "upper": np.minimum((np.arange(nb) + 1) * width, 1.0),
            "n_items": n,
            "share_of_items": n / total,
            "accuracy": np.where(n > 0, hits / np.maximum(n, 1), np.nan),
            "mean_confidence": pd.Series(conf).groupby(idx).mean().reindex(range(nb)).to_numpy(),
            "cumulative_coverage": cum_n / total,
            "cumulative_accuracy": np.where(cum_n > 0, cum_hits / np.maximum(cum_n, 1), np.nan),
        })
    return tbl


def bucket_tables(pred: pd.DataFrame, level_columns: Sequence[str], width: float = 0.1) -> pd.DataFrame:
    return pd.concat([bucket_table(pred, level_columns, d, width)
                      for d in range(1, len(level_columns) + 1)], ignore_index=True)


def split_bucket_accuracy(pred: pd.DataFrame, level_columns: Sequence[str], depth: int,
                          width: float, split_col: str = "fold") -> pd.DataFrame:
    """Accuracy per (split, bucket): the spread across folds/repeats."""
    conf = pred[f"L{depth} Confidence"].to_numpy(dtype=float)
    ok = correct_flags(pred, level_columns)[f"L{depth} Correct"].to_numpy().astype(float)
    frame = pd.DataFrame({"split": pred[split_col].to_numpy(),
                          "bucket": _bucket_index(conf, width), "ok": ok})
    g = frame.groupby(["split", "bucket"])["ok"].agg(["mean", "size"]).reset_index()
    g["bucket_label"] = [ _bucket_labels(width)[i] for i in g["bucket"] ]
    return g.rename(columns={"mean": "accuracy", "size": "n_items"})


def headline(pred: pd.DataFrame, level_columns: Sequence[str], split_col: Optional[str] = None) -> pd.DataFrame:
    """Accuracy by level, overall and (optionally) per split, with mean/std across splits."""
    rows = [{"split": "all", **accuracy_by_level(pred, level_columns)}]
    if split_col and split_col in pred.columns:
        per = [{"split": s, **accuracy_by_level(g, level_columns)}
               for s, g in pred.groupby(split_col, sort=True)]
        rows = per + rows
        per_df = pd.DataFrame(per)
        cols = [c for c in per_df.columns if c != "split"]
        rows.append({"split": "std across splits", **per_df[cols].std(ddof=1).to_dict()})
    return pd.DataFrame(rows).set_index("split")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _grid(n: int):
    import matplotlib.pyplot as plt

    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.2 * rows), squeeze=False)
    return fig, axes.ravel()


BAR_COLOR = "steelblue"
LINE_COLOR = "firebrick"


def _accept_boundary(tbl: pd.DataFrame, target_accuracy: Optional[float]):
    """(x position, label) of the dashed green line, or None.

    As in NB-Category-Classifier: the lowest bucket whose cumulative accuracy
    (reading from the top bucket down) still meets target_accuracy. The line
    sits at that bucket's left edge, so every bar to its right is included.
    """
    if target_accuracy is None:
        return None
    ok = np.flatnonzero(tbl["cumulative_accuracy"].to_numpy() >= target_accuracy)
    if not len(ok):
        return None
    j = int(ok[0])
    cov = tbl["cumulative_coverage"].iloc[j]
    return j - 0.5, f"{target_accuracy:.0%} at {cov:.1%} coverage"


def _draw_accuracy_by_bucket(ax1, tbl: pd.DataFrame, annotate: bool = True,
                             target_accuracy: Optional[float] = None):
    """Item-count bars against an accuracy line (port of NB-Category-Classifier's chart)."""
    x = tbl["bucket"].astype(str)
    ax1.bar(x, tbl["n_items"], color=BAR_COLOR, alpha=0.7)
    ax1.set_xlabel("Probability bucket")
    ax1.set_ylabel("Number of items", color=BAR_COLOR)
    ax1.tick_params(axis="y", labelcolor=BAR_COLOR)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, tbl["accuracy"], color=LINE_COLOR, marker="o", linewidth=2)
    ax2.set_ylabel("Path accuracy", color=LINE_COLOR)
    ax2.tick_params(axis="y", labelcolor=LINE_COLOR)
    ax2.set_ylim(0, 1.05)

    if annotate:
        for xi, yi in zip(x, tbl["accuracy"]):
            if pd.notna(yi):
                ax2.annotate(f"{yi:.1%}", (xi, yi), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9, color=LINE_COLOR)
        total = tbl["n_items"].sum()
        if total:
            for xi, n in zip(x, tbl["n_items"]):
                ax1.annotate(f"{n / total:.1%}", (xi, 0), textcoords="offset points",
                             xytext=(0, 3), ha="center", va="bottom", fontsize=9, color="black")

    if target_accuracy is not None:
        ax2.axhline(target_accuracy, linestyle="--", color="gray", linewidth=1)

    boundary = _accept_boundary(tbl, target_accuracy)
    if boundary is not None:
        pos, label = boundary
        ax1.axvline(pos, linestyle="--", color="darkgreen", linewidth=1.2)
        ax1.annotate(label, (pos, ax1.get_ylim()[1]), textcoords="offset points",
                     xytext=(3, -10), fontsize=8, color="darkgreen", rotation=90, va="top")
    return ax2


def plot_accuracy_by_bucket(pred, level_columns, width=0.1, depth: Optional[int] = None,
                            target_accuracy: Optional[float] = 0.95):
    """Item count (bars) and accuracy (line) by confidence bucket, with the dashed
    green line marking the coverage at which cumulative accuracy meets target_accuracy.

    depth=<d>  one full-size chart for that level, with % labels (as the old repo's
               plot_accuracy_by_bucket);
    depth=None one panel per level, labels off (as the old plot_accuracy_by_bucket_grid).
    """
    import matplotlib.pyplot as plt

    if depth:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        _draw_accuracy_by_bucket(ax1, bucket_table(pred, level_columns, depth, width),
                                 annotate=True, target_accuracy=target_accuracy)
        ax1.set_title(f"Path accuracy and item count by probability bucket ({level_columns[depth - 1]})")
        fig.tight_layout()
        return fig

    depths = list(range(1, len(level_columns) + 1))
    ncols = 2
    nrows = int(np.ceil(len(depths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.6 * nrows), squeeze=False)
    for i, d in enumerate(depths):
        ax = axes[i // ncols][i % ncols]
        _draw_accuracy_by_bucket(ax, bucket_table(pred, level_columns, d, width),
                                 annotate=False, target_accuracy=target_accuracy)
        title = f"Correct through {level_columns[d - 1]}"
        if target_accuracy is not None:
            title += f"  (target {target_accuracy:.1%})"
        ax.set_title(title, fontsize=10)
    for i in range(len(depths), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle("Accuracy and item count by probability bucket, per level", y=1.0)
    fig.tight_layout()
    return fig


def plot_violins(pred, level_columns, width=0.1, depth: Optional[int] = None, split_col: str = "fold"):
    """Distribution across folds/repeats of the accuracy in each confidence bucket."""
    depths = [depth] if depth else list(range(1, len(level_columns) + 1))
    labels = _bucket_labels(width)
    fig, axes = _grid(len(depths))
    for ax, d in zip(axes, depths):
        g = split_bucket_accuracy(pred, level_columns, d, width, split_col)
        data, pos = [], []
        for b in range(len(labels)):
            vals = g.loc[g["bucket"] == b, "accuracy"].dropna().to_numpy()
            if len(vals):
                data.append(vals)
                pos.append(b)
        if data:
            multi = [i for i, v in enumerate(data) if len(v) > 1 and np.ptp(v) > 0]
            if multi:
                ax.violinplot([data[i] for i in multi], positions=[pos[i] for i in multi],
                              widths=0.8, showmeans=True, showextrema=True)
            for p, v in zip(pos, data):
                ax.scatter(np.full(len(v), p), v, s=8, color="black", alpha=0.6, zorder=3)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"accuracy per {split_col}")
        ax.set_title(f"L{d} ({level_columns[d - 1]}): accuracy by bucket across {split_col}s")
    for ax in axes[len(depths):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_coverage(pred, level_columns, depth: Optional[int] = None):
    """Accuracy of the items kept vs share of items kept, keeping the most
    confident first (one curve per level, or one subplot per level)."""
    depths = [depth] if depth else list(range(1, len(level_columns) + 1))
    flags = correct_flags(pred, level_columns)
    fig, axes = _grid(len(depths))
    for ax, d in zip(axes, depths):
        conf = pred[f"L{d} Confidence"].to_numpy(dtype=float)
        ok = flags[f"L{d} Correct"].to_numpy().astype(float)
        order = np.argsort(-conf, kind="mergesort")
        cum_acc = np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)
        coverage = np.arange(1, len(ok) + 1) / max(len(ok), 1)
        step = max(1, len(ok) // 2000)
        ax.plot(coverage[::step], cum_acc[::step], color="#2c3e50")
        for target in (0.9, 0.95):
            hit = np.flatnonzero(cum_acc >= target)
            if len(hit):
                cov = coverage[hit[-1]]
                ax.axhline(target, ls=":", color="grey", lw=0.8)
                ax.annotate(f"{target:.0%} acc @ {cov:.0%} coverage", (cov, target),
                            fontsize=7, xytext=(5, -12), textcoords="offset points")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("coverage (share of items, most confident first)")
        ax.set_ylabel("accuracy of covered items")
        ax.set_title(f"L{d} ({level_columns[d - 1]}): coverage curve")
    for ax in axes[len(depths):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def save_figure(fig, folder, name: str, tracker=None) -> None:
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=120, bbox_inches="tight")
    if tracker is not None:
        tracker.log_figure(fig, f"plots/{name}.png")


def plot_training_history(history_df: pd.DataFrame, title: str = "Training history"):
    """Loss (train and validation) and validation accuracy per level, by epoch.
    (Used by the NN repo; history_df is History.to_frame().)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="train loss", color=BAR_COLOR)
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="validation loss", color=LINE_COLOR)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    for c in [c for c in history_df.columns if c.startswith("val_acc")]:
        ax2.plot(history_df["epoch"], history_df[c], marker=".", label=c.replace("val_acc ", ""))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation accuracy (head argmax)")
    ax2.set_ylim(0, 1.02)
    ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
