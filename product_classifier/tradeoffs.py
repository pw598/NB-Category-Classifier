"""What two data-shrinking decisions actually cost.

Both settings covered here make accuracy look better by giving the model
less to be wrong about, so neither can be judged on accuracy alone:

    min_path_count      drops rare categories. Accuracy rises; the dropped
                        products still have to be classified by someone.
    max_features        caps the vocabulary. A smaller vocabulary means
                        more descriptions the model cannot read at all.

The useful figure in both cases is marginal, not average. "Accuracy went
from 91% to 89%" says little. "Lowering the threshold keeps 5,000 more
products and adds 900 errors, so the products added back are classified
at 82%" is a decision.

Two things make the two sweeps read differently, and it matters:

* Varying min_path_count changes *which rows exist*, so accuracy across
  settings is measured on different populations and is not strictly
  comparable. Absolute error counts are, which is why they are reported.
* Varying max_features leaves the row set alone, so accuracy is directly
  comparable and the marginal figure is simply errors avoided.

Runtime is (settings x n_splits) model fits. Start with n_splits=2 and a
sample_frac while finding the interesting range.
"""

from __future__ import annotations

import contextlib
import gc
import io
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data as data_mod
from .config import PipelineConfig
from .model import HierarchicalPathClassifier


# ---------------------------------------------------------------------------
# Shared scoring
# ---------------------------------------------------------------------------

def _quiet(enabled: bool):
    return contextlib.redirect_stdout(io.StringIO()) if enabled else contextlib.nullcontext()


def _prepare(cfg, df, levels, min_path_count, sample_frac, random_state, verbose):
    frame = data_mod.drop_null_rows(
        df, cfg.data.text_col, levels,
        cfg.data.drop_null_text, cfg.data.drop_null_labels, verbose=False,
    )
    if sample_frac is not None:
        frame = (
            frame.groupby(levels[-1], group_keys=False, observed=True)
            .apply(lambda g: g.sample(frac=sample_frac, random_state=random_state))
        )
    frame = data_mod.drop_rare_paths(
        frame, levels, min_path_count, verbose=False
    ).reset_index(drop=True)
    counts = frame[levels[-1]].astype(str).value_counts()
    if len(frame) == 0 or counts.min() < 2:
        return None
    return frame


def _score(cfg, frame, levels, n_splits, test_size, random_state, quiet=True,
           keep_model=False):
    """Repeated-holdout accuracy at full depth, and optionally the last model.

    Accuracy is pooled over folds -- total correct over total tested --
    rather than averaged, so folds of unequal size cannot distort it.

    keep_model exists only for the vocabulary sweep, which needs the fitted
    vectorizer afterwards. The rare-path sweep does not, and a fitted model
    at a low threshold is several gigabytes, so it is freed inside the loop
    rather than surviving until the caller gets round to deleting it.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )
    y = frame[levels[-1]].astype(str).to_numpy()
    n_correct = n_tested = 0
    clf = None

    for k, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(frame)), y)
    ):
        with _quiet(quiet):
            clf = HierarchicalPathClassifier(level_columns=levels, config=cfg.model)
            clf.fit(frame.loc[train_idx, cfg.data.text_col],
                    frame.loc[train_idx, levels])
            pred = clf.predict(frame.loc[test_idx, cfg.data.text_col])
        truth = frame.loc[test_idx, levels].astype(str).to_numpy()
        n_correct += int((pred[levels].astype(str).to_numpy() == truth).all(axis=1).sum())
        n_tested += len(test_idx)
        del pred, truth
        if keep_model and k == n_splits - 1:
            gc.collect()
        else:
            del clf
            clf = None
            gc.collect()

    return {"accuracy": n_correct / n_tested if n_tested else np.nan,
            "n_tested": n_tested}, clf


# ---------------------------------------------------------------------------
# Pre-flight sizing
# ---------------------------------------------------------------------------

def estimate_memory(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    thresholds: Sequence[int] = (50, 20, 10, 5, 2),
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Projected peak memory per threshold, without fitting anything.

    Only the label columns are needed to count classes and paths, so this
    is fast and can be run before committing to a sweep that might not
    fit. Three allocations dominate, all linear in the class count, which
    is why the lowest threshold is the one that fails:

        resident   2 x n_classes x n_features x 8 B, summed over levels
                   (MultinomialNB keeps feature_count_ and
                   feature_log_prob_ dense, in float64, for every level at
                   once)
        fit        fit_chunk_size x max(n_classes) x 8 B
                   (sklearn densifies the labels; bounded by the chunk
                   size, which is what makes chunked fitting worth it)
        scoring    batch_size x n_paths x 4 B
                   (the joint score matrix, float32)

    n_features is taken from max_features, so these are upper bounds --
    min_df usually leaves the fitted vocabulary smaller. Treat the numbers
    as a ranking and an order of magnitude, not a promise.
    """
    levels = cfg.active_levels()
    n_features = cfg.model.vectorizer.max_features
    if n_features is None:
        raise ValueError(
            "max_features is None, so the vocabulary size cannot be bounded "
            "ahead of fitting. Set a cap to estimate."
        )

    rows = []
    for t in sorted(thresholds, reverse=True):
        frame = _prepare(cfg, df, levels, t, sample_frac, random_state, False)
        if frame is None:
            rows.append({"min_path_count": t, "usable": False})
            continue
        per_level = {lv: frame[lv].astype(str).nunique() for lv in levels}
        n_paths = frame[levels].drop_duplicates().shape[0]

        resident = sum(2 * c * n_features * 8 for c in per_level.values())
        fit_peak = (cfg.model.fit_chunk_size or len(frame)) * max(per_level.values()) * 8
        score_peak = cfg.model.batch_size * n_paths * 4

        rows.append({
            "min_path_count": t,
            "usable": True,
            "rows_kept": len(frame),
            "classes_deepest": per_level[levels[-1]],
            "distinct_paths": n_paths,
            "resident_gb": resident / 1e9,
            "fit_peak_gb": fit_peak / 1e9,
            "scoring_peak_gb": score_peak / 1e9,
            "projected_peak_gb": (resident + max(fit_peak, score_peak)) / 1e9,
        })
        del frame
        gc.collect()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rare-path threshold
# ---------------------------------------------------------------------------

def sweep_min_path_count(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    thresholds: Sequence[int] = (50, 20, 10, 5, 2),
    n_splits: int = 3,
    test_size: float = 0.25,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Accuracy and error count as the rare-path threshold is lowered.

    checkpoint_path writes the table after every threshold completes, and
    with resume=True any threshold already in that file is skipped. The
    lowest thresholds are the ones that exhaust memory, and they run last,
    so without this a crash discards every threshold that had already
    succeeded.

    Thresholds are processed strictest first, so each row's incremental
    columns describe what *lowering* the threshold to that value added.

    Columns:
        rows_kept              products surviving the filter
        distinct_paths         categories surviving
        accuracy               full-path accuracy, on that row set
        est_errors             rows_kept x (1 - accuracy): errors expected
                               across the retained catalogue, which is the
                               figure that stays comparable when accuracy
                               is not
        rows_added             extra products kept versus the row above
        errors_added           extra expected errors versus the row above
        marginal_accuracy      1 - errors_added / rows_added: how well the
                               products just added back get classified.
                               The number to decide on -- it isolates the
                               categories the change actually admits,
                               instead of averaging them with the easy
                               ones that were never at risk.
        marginal_reliable      False when marginal_accuracy falls outside
                               0-1, which it can: est_errors uses accuracy
                               over the *whole* retained set, so if that
                               accuracy happens to rise when rare
                               categories are added, errors_added goes
                               negative and the ratio stops meaning
                               anything. Normally a sign of too few folds
                               or too small a step between thresholds
                               rather than a real effect -- raise n_splits
                               before believing it.
    """
    levels = cfg.active_levels()
    rows = []
    done = set()

    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        prior = pd.read_csv(checkpoint_path)
        base = [c for c in ("min_path_count", "rows_kept", "distinct_paths",
                            "accuracy", "est_errors") if c in prior.columns]
        rows = prior[base].to_dict("records")
        done = set(prior["min_path_count"].astype(int))
        if verbose and done:
            print(f"[rare] resuming; already done: {sorted(done, reverse=True)}")

    for t in sorted(thresholds, reverse=True):
        if int(t) in done:
            continue
        frame = _prepare(cfg, df, levels, t, sample_frac, random_state, verbose)
        if frame is None:
            if verbose:
                print(f"[rare] threshold {t}: unusable (empty, or a category "
                      "left with one member)")
            continue
        scored, _ = _score(cfg, frame, levels, n_splits, test_size, random_state,
                           keep_model=False)
        rows.append({
            "min_path_count": t,
            "rows_kept": len(frame),
            "distinct_paths": frame[levels].drop_duplicates().shape[0],
            "accuracy": scored["accuracy"],
            "est_errors": len(frame) * (1 - scored["accuracy"]),
        })
        if verbose:
            r = rows[-1]
            print(f"[rare] threshold {t:>3}: {r['rows_kept']:>7,} rows, "
                  f"{r['distinct_paths']:>5,} paths, accuracy {r['accuracy']:.4f}")

        del frame
        gc.collect()
        if checkpoint_path:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return add_marginals(out, verbose=verbose)


def add_marginals(results: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Recompute the incremental columns over a whole table.

    Needed when the sweep has been run one threshold at a time and the
    pieces concatenated: diffs only mean something once every threshold is
    present and ordered strictest first.
    """
    out = results.sort_values("min_path_count", ascending=False).reset_index(drop=True)
    out["rows_added"] = out["rows_kept"].diff()
    out["errors_added"] = out["est_errors"].diff()
    out["marginal_accuracy"] = 1 - out["errors_added"] / out["rows_added"].replace(0, np.nan)
    # object dtype so the first row can hold NA alongside real booleans
    reliable = out["marginal_accuracy"].between(0.0, 1.0).astype(object)
    reliable[out["marginal_accuracy"].isna()] = pd.NA
    out["marginal_reliable"] = reliable

    if verbose and (out["marginal_reliable"] == False).any():  # noqa: E712  # noqa
        bad = out.loc[out["marginal_reliable"] == False, "min_path_count"].tolist()  # noqa: E712
        print(f"[rare] marginal accuracy out of range at threshold(s) {bad}: "
              "accuracy moved the unexpected direction, so the marginal "
              "figure is noise. Raise n_splits or widen the steps.")
    return out


# ---------------------------------------------------------------------------
# Vectorizer size
# ---------------------------------------------------------------------------

def sweep_max_features(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    sizes: Sequence[Optional[int]] = (5_000, 10_000, 20_000, 50_000, 100_000),
    n_splits: int = 3,
    test_size: float = 0.25,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Accuracy, error count and unreadable descriptions as vocabulary grows.

    The row set is identical at every size, so accuracy here *is* directly
    comparable and errors_avoided is a clean marginal figure.

    Pass None among the sizes for "no cap", which is the ceiling any
    smaller vocabulary is being compared against.

    Returns (results, oov_frames) where oov_frames maps each size to the
    descriptions that vocabulary cannot read at all -- every word unknown.
    Those rows still receive a confident prediction, drawn from category
    frequencies rather than from the product, so they are the concrete
    cost of a smaller vocabulary.

    Columns:
        vocabulary_size        features actually fitted, which is below the
                               cap once min_df has done its work
        n_fully_oov            descriptions with no readable word
        mean_oov_rate          share of words unknown, across the file
        accuracy, est_errors   as in the rare-path sweep
        errors_avoided         est_errors at the previous, smaller size
                               minus this one -- what the extra vocabulary
                               bought
    """
    import copy

    from .quality import describe_inputs

    levels = cfg.active_levels()
    frame = _prepare(cfg, df, levels, cfg.split.min_path_count,
                     sample_frac, random_state, verbose)
    if frame is None:
        raise ValueError("No usable rows after filtering; check min_path_count.")
    if verbose:
        print(f"[vocab] {len(frame):,} rows held constant across all sizes")

    ordered = sorted(sizes, key=lambda s: (s is None, s))
    rows, oov_frames = [], {}

    for size in ordered:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.model.vectorizer.max_features = size
        scored, clf = _score(run_cfg, frame, levels, n_splits, test_size, random_state,
                             keep_model=True)

        quality = describe_inputs(clf, df[cfg.data.text_col])
        fully = ((quality["n_tokens"] > 0) & (quality["n_known"] == 0)).to_numpy()
        keep = [c for c in ([cfg.data.id_col] if cfg.data.id_col else [])
                + [cfg.data.text_col] + levels if c in df.columns]
        oov_frames[size] = pd.concat(
            [df.loc[fully, keep].reset_index(drop=True),
             quality.loc[fully, ["n_tokens", "n_known", "oov_rate"]]
             .reset_index(drop=True)],
            axis=1,
        )

        rows.append({
            "max_features": size,
            "vocabulary_size": len(clf.vectorizer.vocabulary_),
            "n_fully_oov": int(fully.sum()),
            "share_fully_oov": float(fully.mean()),
            "mean_oov_rate": float(quality["oov_rate"].mean()),
            "accuracy": scored["accuracy"],
            "est_errors": len(frame) * (1 - scored["accuracy"]),
        })
        if verbose:
            r = rows[-1]
            print(f"[vocab] max_features={str(size):>7}: vocab "
                  f"{r['vocabulary_size']:>7,}, unreadable {r['n_fully_oov']:>6,}, "
                  f"accuracy {r['accuracy']:.4f}")
        del clf, quality
        gc.collect()

    out = pd.DataFrame(rows)
    out["errors_avoided"] = -out["est_errors"].diff()
    return out, oov_frames


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_min_path_tradeoff(results: pd.DataFrame, figsize=(11, 5)):
    """Products retained against accuracy, with the marginal figure beside it.

    The left panel is the headline trade; the right is the one to decide
    on, since it prices only the categories each step actually admits.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=figsize)
    x = results["min_path_count"].astype(str)

    ax1.bar(x, results["rows_kept"], color="steelblue", alpha=0.7)
    ax1.set_xlabel("min_path_count (strictest first)")
    ax1.set_ylabel("Products retained", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(x, results["accuracy"], color="firebrick", marker="o")
    ax2.set_ylabel("Full-path accuracy", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax1.set_title("Retention vs. accuracy")

    ax3.plot(x, results["marginal_accuracy"], color="darkgreen", marker="s")
    ax3.axhline(0, color="gray", linewidth=0.8)
    for xi, yi, n in zip(x, results["marginal_accuracy"], results["rows_added"]):
        if pd.notna(yi):
            ax3.annotate(f"+{int(n):,}", (xi, yi), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    ax3.set_xlabel("min_path_count")
    ax3.set_ylabel("Accuracy on the products added back")
    ax3.set_title("Marginal accuracy of each step down")
    fig.tight_layout()
    return fig


def plot_max_features_tradeoff(results: pd.DataFrame, figsize=(11, 5)):
    """Unreadable descriptions against accuracy, as vocabulary grows."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=figsize)
    x = results["max_features"].astype(str)

    ax1.bar(x, results["n_fully_oov"], color="steelblue", alpha=0.7)
    ax1.set_xlabel("max_features")
    ax1.set_ylabel("Descriptions with no readable word", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(x, results["accuracy"], color="firebrick", marker="o")
    ax2.set_ylabel("Full-path accuracy", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax1.set_title("Unreadable descriptions vs. accuracy")

    ax3.plot(results["vocabulary_size"], results["accuracy"],
             color="darkgreen", marker="s")
    for _, r in results.iterrows():
        ax3.annotate(str(r["max_features"]),
                     (r["vocabulary_size"], r["accuracy"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax3.set_xlabel("Vocabulary actually fitted")
    ax3.set_ylabel("Full-path accuracy")
    ax3.set_title("Diminishing returns")
    fig.tight_layout()
    return fig
