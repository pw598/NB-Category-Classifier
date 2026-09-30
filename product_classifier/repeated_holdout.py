"""Repeated 75/25 holdout, with per-fold calibration buckets.

Five random splits rather than a five-way partition. The distinction
matters: a partition gives each row exactly one out-of-fold prediction,
while repeated holdout re-draws the test set each time, so rows can appear
in several test sets or none. What it buys is five independent estimates of
accuracy *within each confidence bucket*, which is what a spread across
folds needs -- a partition gives only one estimate per row and so no
spread to measure.

Returns everything downstream steps need from one pass, since the model
fits are the expensive part:

    buckets   accuracy per confidence bucket per fold  -> mean, std, violins
    prefix    pooled prefix probabilities, all test rows from all folds
    actual    the true levels, aligned to prefix
    folds     fold id per pooled row, for cross-fitting thresholds
    frame     pooled source rows, for joining ids onto per-item output
    last_clf  the final fold's fitted classifier, for its vocabulary

Note the pooling: `prefix` is the concatenation of five test sets, so it is
longer than the input and a row may appear more than once. That is correct
for fitting thresholds -- every entry is a genuine held-out prediction --
but it means row counts in `prefix` are not catalogue counts.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from . import data as data_mod
from .config import PipelineConfig
from .model import PROBABILITY_COL, HierarchicalPathClassifier

PREFIX_PROB_FMT = "L{d} Prefix Probability"


@dataclass
class HoldoutResult:
    buckets: pd.DataFrame
    prefix: pd.DataFrame
    actual: pd.DataFrame
    folds: np.ndarray
    frame: pd.DataFrame
    last_clf: HierarchicalPathClassifier
    dropped: pd.DataFrame = None


def _bucket_edges(
    scores: np.ndarray, bucket_width: float, n_quantile_buckets: Optional[int]
) -> np.ndarray:
    """Uniform edges, or quantile edges when asked.

    Raw (unnormalised) probabilities bunch up, so uniform bands can leave
    one bucket holding almost everything and the rest nearly empty. Quantile
    edges put the same number of products in every band, which is what you
    want when each row of the table has to support a decision.
    """
    if n_quantile_buckets:
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_quantile_buckets + 1)))
        if len(edges) < 3:
            raise ValueError(
                "Scores are too concentrated for quantile buckets; "
                "use bucket_width instead."
            )
        return edges
    return np.arange(0, 1.0 + bucket_width / 2, bucket_width)


def repeated_holdout(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.25,
    bucket_width: float = 0.05,
    n_quantile_buckets: Optional[int] = None,
    min_path_count: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> HoldoutResult:
    """Fit and score n_splits times on independent random holdouts.

    Scores at full depth, on whatever probability scale
    cfg.model.normalise_probabilities selects.

    min_path_count overrides cfg.split.min_path_count for this call, so
    the rare-path threshold can be varied without editing the config.
    Raising it removes categories the model has little evidence for --
    which usually lifts accuracy while quietly shrinking the catalogue the
    model is willing to touch, so the count of what it discards is
    returned in .dropped rather than only printed.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    levels = cfg.active_levels()
    threshold = cfg.split.min_path_count if min_path_count is None else min_path_count

    n_start = len(df)
    paths_start = df[levels].drop_duplicates().shape[0]

    frame = data_mod.drop_null_rows(
        df, cfg.data.text_col, levels,
        cfg.data.drop_null_text, cfg.data.drop_null_labels, verbose=verbose,
    )
    n_after_null = len(frame)

    frame = data_mod.drop_rare_paths(
        frame, levels, threshold, verbose=verbose
    ).reset_index(drop=True)
    n_after_rare = len(frame)
    paths_end = frame[levels].drop_duplicates().shape[0]

    counts = frame[levels[-1]].astype(str).value_counts()
    if counts.min() < 2:
        raise ValueError(
            f"{int((counts < 2).sum())} categories at '{levels[-1]}' have a "
            "single member, so a stratified split is impossible. Raise "
            "min_path_count to at least 2."
        )

    dropped = pd.DataFrame(
        [
            {"stage": "loaded", "rows": n_start,
             "rows_dropped": 0, "share_dropped": 0.0,
             "distinct_paths": paths_start},
            {"stage": "after dropping null text/labels", "rows": n_after_null,
             "rows_dropped": n_start - n_after_null,
             "share_dropped": (n_start - n_after_null) / n_start if n_start else np.nan,
             "distinct_paths": np.nan},
            {"stage": f"after dropping paths seen < {threshold} times",
             "rows": n_after_rare,
             "rows_dropped": n_after_null - n_after_rare,
             "share_dropped": (n_after_null - n_after_rare) / n_start if n_start else np.nan,
             "distinct_paths": paths_end},
        ]
    )
    dropped.attrs["min_path_count"] = threshold
    dropped.attrs["paths_dropped"] = int(paths_start - paths_end)
    dropped.attrs["rows_kept"] = n_after_rare
    dropped.attrs["share_kept"] = n_after_rare / n_start if n_start else np.nan

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )
    y = frame[levels[-1]].astype(str).to_numpy()

    bucket_rows, prefix_parts, actual_parts, frame_parts, fold_ids = [], [], [], [], []
    clf = None

    for k, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(frame)), y)):
        if verbose:
            print(f"[holdout] fold {k + 1}/{n_splits}: "
                  f"train {len(train_idx):,}, test {len(test_idx):,}")
        clf = HierarchicalPathClassifier(level_columns=levels, config=cfg.model)
        clf.fit(frame.loc[train_idx, cfg.data.text_col], frame.loc[train_idx, levels])

        test_texts = frame.loc[test_idx, cfg.data.text_col]
        truth = frame.loc[test_idx, levels].astype(str).reset_index(drop=True)

        pred = clf.predict(test_texts)
        correct = (
            pred[levels].astype(str).to_numpy() == truth.to_numpy()
        ).all(axis=1)
        scores = pred[PROBABILITY_COL].to_numpy(dtype=float)

        edges = _bucket_edges(scores, bucket_width, n_quantile_buckets)
        band = pd.cut(scores, bins=edges, include_lowest=True)
        per = pd.DataFrame({"bucket": band, "correct": correct}).groupby(
            "bucket", observed=False
        ).agg(n_items=("correct", "size"), accuracy=("correct", "mean")).reset_index()
        per["fold"] = k
        bucket_rows.append(per)

        prefix_parts.append(clf.predict_prefix_probabilities(test_texts))
        actual_parts.append(truth)
        frame_parts.append(frame.loc[test_idx].reset_index(drop=True))
        fold_ids.append(np.full(len(test_idx), k))

        if k < n_splits - 1:          # keep the last one for its vocabulary
            del clf
            clf = None
            gc.collect()

    buckets = pd.concat(bucket_rows, ignore_index=True)
    buckets["bucket_label"] = buckets["bucket"].astype(str)
    return HoldoutResult(
        buckets=buckets,
        prefix=pd.concat(prefix_parts, ignore_index=True),
        actual=pd.concat(actual_parts, ignore_index=True),
        folds=np.concatenate(fold_ids),
        frame=pd.concat(frame_parts, ignore_index=True),
        last_clf=clf,
        dropped=dropped,
    )


def bucket_accuracy_by_level(
    prefix: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    folds: np.ndarray,
    bucket_width: float = 0.05,
    n_quantile_buckets: Optional[int] = None,
) -> pd.DataFrame:
    """Per-fold bucket accuracy at every depth, not just the full path.

    Computed from the pooled predictions repeated_holdout already returned,
    so this costs nothing beyond arithmetic -- no refitting.

    At depth d the score is that depth's prefix probability and "correct"
    means the whole prefix down to d is right. Each depth is bucketed on
    its own distribution, which matters on the raw scale: a depth-1 score
    and a depth-4 score are not the same kind of number, so shared bucket
    edges would put them in incomparable bands.
    """
    from .cascade import prefix_correct

    rows = []
    for d in range(1, len(level_columns) + 1):
        scores = prefix[PREFIX_PROB_FMT.format(d=d)].to_numpy(dtype=float)
        correct = prefix_correct(prefix, actual, level_columns, d)
        edges = _bucket_edges(scores, bucket_width, n_quantile_buckets)
        band = pd.cut(scores, bins=edges, include_lowest=True)

        per = (
            pd.DataFrame({"bucket": band, "correct": correct, "fold": folds})
            .groupby(["fold", "bucket"], observed=False)
            .agg(n_items=("correct", "size"), accuracy=("correct", "mean"))
            .reset_index()
        )
        per["depth"] = d
        rows.append(per)

    out = pd.concat(rows, ignore_index=True)
    out["bucket_label"] = out["bucket"].astype(str)
    return out


def bucket_stats(buckets: pd.DataFrame, group_extra: tuple = ()) -> pd.DataFrame:
    """Mean and standard deviation of accuracy per bucket, across folds.

    Pass group_extra=("depth",) for a frame from bucket_accuracy_by_level,
    so each depth is summarised separately.

    n_folds_present is worth reading alongside: a bucket seen in two folds
    has a standard deviation, but not one that means anything.
    """
    keys = list(group_extra) + ["bucket_label"]
    out = (
        buckets.dropna(subset=["accuracy"])
        .groupby(keys, observed=True)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
            n_folds_present=("accuracy", "size"),
            mean_n_items=("n_items", "mean"),
        )
        .reset_index()
    )
    order = (
        buckets.drop_duplicates(keys)
        .set_index(keys)["bucket"]
        .apply(lambda b: b.left if hasattr(b, "left") else np.nan)
    )
    out["_left"] = list(order.reindex(pd.MultiIndex.from_frame(out[keys])
                                      if len(keys) > 1 else out[keys[0]]))
    return (
        out.sort_values(list(group_extra) + ["_left"])
        .drop(columns="_left")
        .reset_index(drop=True)
    )


def plot_bucket_violins(buckets: pd.DataFrame, figsize=(11, 5)):
    """Distribution of accuracy across folds, one violin per bucket.

    Buckets present in fewer than three folds are drawn as points instead:
    a violin over two numbers implies a shape the data cannot support.
    """
    import matplotlib.pyplot as plt

    stats = bucket_stats(buckets)
    labels = stats["bucket_label"].tolist()
    grouped = {
        lab: buckets.loc[
            (buckets["bucket_label"] == lab) & buckets["accuracy"].notna(), "accuracy"
        ].to_numpy()
        for lab in labels
    }

    fig, ax = plt.subplots(figsize=figsize)
    violin_pos = [i for i, lab in enumerate(labels) if len(grouped[lab]) >= 3]
    if violin_pos:
        ax.violinplot([grouped[labels[i]] for i in violin_pos],
                      positions=violin_pos, showmeans=True, widths=0.7)
    for i, lab in enumerate(labels):
        vals = grouped[lab]
        ax.scatter(np.full(len(vals), i), vals, s=18, zorder=3,
                   color="#333333", alpha=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("confidence bucket (raw probability)")
    ax.set_title("Accuracy per confidence bucket, across folds")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def level_bucket_summary(buckets: pd.DataFrame) -> dict:
    """Turn the per-fold long frame into one bucket_summary per depth.

    The result has the columns the plotting functions expect, so
    {depth: frame} can be handed straight to plot_accuracy_by_bucket_grid
    and plot_coverage_curve_grid, or an individual frame to the existing
    single-chart functions.

    Accuracy is pooled across folds -- total correct over total items --
    rather than the unweighted mean of the fold accuracies. The two agree
    closely when folds are the same size, but only the pooled figure makes
    the cumulative columns arithmetically consistent, since those are
    running totals over items. The unweighted version is kept alongside as
    mean_fold_accuracy for anyone comparing against the violins.
    """
    out = {}
    for depth, grp in buckets.groupby("depth", observed=True):
        per = (
            grp.groupby("bucket_label", observed=True)
            .agg(
                n_items=("n_items", "sum"),
                _weighted=("accuracy", lambda s: np.nan),
                mean_fold_accuracy=("accuracy", "mean"),
                _left=("bucket", lambda s: s.iloc[0].left
                       if hasattr(s.iloc[0], "left") else np.nan),
            )
            .reset_index()
            .drop(columns="_weighted")
        )
        # Pooled accuracy: sum of (items x accuracy) over sum of items.
        num = (
            grp.assign(_c=grp["n_items"] * grp["accuracy"].fillna(0))
            .groupby("bucket_label", observed=True)["_c"].sum()
        )
        den = grp.groupby("bucket_label", observed=True)["n_items"].sum()
        per["accuracy"] = (num / den.replace(0, np.nan)).reindex(
            per["bucket_label"]
        ).to_numpy()

        per["_right"] = (
            grp.groupby("bucket_label", observed=True)["bucket"]
            .apply(lambda s: s.iloc[0].right if hasattr(s.iloc[0], "right") else np.nan)
            .reindex(per["bucket_label"]).to_numpy()
        )
        per = per.sort_values("_left").reset_index(drop=True)
        # Numeric edges kept so a threshold can be located among the bars;
        # the label alone is a string and cannot be compared against a score.
        per = per.rename(columns={"bucket_label": "Probability Bucket",
                                  "_left": "bucket_left", "_right": "bucket_right"})

        total = per["n_items"].sum()
        per["share_of_items"] = per["n_items"] / total if total else np.nan
        rev = per.iloc[::-1]
        cum_n = rev["n_items"].cumsum()
        cum_correct = (rev["n_items"] * rev["accuracy"].fillna(0)).cumsum()
        per["cumulative_coverage"] = (
            (cum_n / total).iloc[::-1].to_numpy() if total else np.nan
        )
        per["cumulative_accuracy"] = (
            (cum_correct / cum_n.replace(0, np.nan)).iloc[::-1].to_numpy()
        )
        out[int(depth)] = per
    return out


def _draw_violins(ax, buckets: pd.DataFrame, labels: List[str], xlabel: str) -> None:
    """Shared drawing routine. Buckets present in fewer than three folds are
    drawn as points only: a violin over two numbers implies a shape the data
    cannot support."""
    grouped = {
        lab: buckets.loc[
            (buckets["bucket_label"] == lab) & buckets["accuracy"].notna(), "accuracy"
        ].to_numpy()
        for lab in labels
    }
    pos = [i for i, lab in enumerate(labels) if len(grouped[lab]) >= 3]
    if pos:
        ax.violinplot([grouped[labels[i]] for i in pos], positions=pos,
                      showmeans=True, widths=0.7)
    for i, lab in enumerate(labels):
        vals = grouped[lab]
        ax.scatter(np.full(len(vals), i), vals, s=14, zorder=3,
                   color="#333333", alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("accuracy")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.3)


def plot_bucket_violins_by_level(
    buckets: pd.DataFrame,
    level_columns: Optional[List[str]] = None,
    ncols: int = 2,
    figsize=None,
    share_y: bool = True,
):
    """One violin panel per level, from bucket_accuracy_by_level().

    Panels are deliberately not forced onto a shared x-axis: each depth is
    bucketed on its own score distribution, so the bands differ between
    panels and pretending otherwise would invite exactly the wrong
    comparison. Read down a panel, not across them.
    """
    import matplotlib.pyplot as plt

    depths = sorted(buckets["depth"].unique())
    nrows = int(np.ceil(len(depths) / ncols))
    figsize = figsize or (7.5 * ncols, 4.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=share_y,
                             squeeze=False)

    stats = bucket_stats(buckets, group_extra=("depth",))
    for n, d in enumerate(depths):
        ax = axes[n // ncols][n % ncols]
        sub = buckets[buckets["depth"] == d]
        labels = stats.loc[stats["depth"] == d, "bucket_label"].tolist()
        _draw_violins(ax, sub, labels, "confidence bucket")
        name = level_columns[d - 1] if level_columns and d <= len(level_columns) else f"depth {d}"
        ax.set_title(f"Correct through {name}", fontsize=10)

    for n in range(len(depths), nrows * ncols):
        axes[n // ncols][n % ncols].axis("off")
    fig.suptitle("Accuracy per confidence bucket, across folds", y=1.0)
    fig.tight_layout()
    return fig


def fully_out_of_vocabulary(
    clf: HierarchicalPathClassifier,
    df: pd.DataFrame,
    text_col: str,
    keep_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Rows whose every word is unknown to the model.

    These still receive a confident-looking prediction, derived entirely
    from how common each category is rather than from the product, so they
    are worth pulling out regardless of what their score says.

    Descriptions with no words at all are reported separately in .attrs,
    since "every word is unknown" is not really true of them.
    """
    from .quality import describe_inputs

    quality = describe_inputs(clf, df[text_col])
    keep = keep_cols or [text_col]
    keep = [c for c in keep if c in df.columns]

    empty = (quality["n_tokens"] == 0).to_numpy()
    fully = ((quality["n_tokens"] > 0) & (quality["n_known"] == 0)).to_numpy()

    out = pd.concat(
        [df.loc[fully, keep].reset_index(drop=True),
         quality.loc[fully, ["n_tokens", "n_known", "oov_rate"]].reset_index(drop=True)],
        axis=1,
    )
    out.attrs["n_rows_checked"] = len(df)
    out.attrs["n_fully_oov"] = int(fully.sum())
    out.attrs["n_empty_description"] = int(empty.sum())
    out.attrs["vocabulary_size"] = len(clf.vectorizer.vocabulary_)
    return out
