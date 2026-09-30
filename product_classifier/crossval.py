"""K-fold cross-validation over the whole catalogue.

A single train/test split leaves three quarters of the data unable to
contribute to evaluation, and only a quarter available for choosing
thresholds. The deep levels feel this most: an L4 threshold is estimated
only from rows confident enough to clear it, which can be a few dozen out
of a 25% slice.

This module removes the split. The model is refitted k times, each fold
predicting the rows it was not trained on, so every product in the file
ends up with a prediction from a model that never saw it. Thresholds are
then chosen from all of them.

    out_of_fold_predictions()   the expensive part: k model refits
    crossfit_accuracy()         Option A thresholds, honestly priced
    crossfit_cost()             Option B thresholds, honestly priced

Cost: k full training runs. At n_levels=4 that is k times the memory
pressure of one run, so start with n_folds=2 while experimenting and move
to 5 for a real answer.

Two layers of folding are going on, and they do different jobs:

* The model-level folds give predictions that are honest with respect to
  *training* -- no row was scored by a model that had seen it.
* Thresholds are then cross-fitted over the same folds, because choosing
  a cut-off is itself an estimate. A threshold picked to look good on a
  set of rows will look better on those rows than on new ones, whatever
  the model did.

Reusing one set of folds for both is deliberate: a row's threshold is
fitted on folds that also trained the model that scored the other rows,
which keeps the two kinds of leakage from crossing.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import data as data_mod
from .cascade import apply_cascade, cascade_report, tune_thresholds
from .cleaning.text import CleaningConfig, clean_dataframe
from .config import PipelineConfig
from .cost_thresholds import cost_optimal_thresholds
from .costing import CostModel, realised_cost
from .model import HierarchicalPathClassifier


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def make_folds(
    labels: pd.Series, n_folds: int = 5, random_state: int = 42, stratify: bool = True
) -> np.ndarray:
    """Fold id per row, stratified on the deepest level by default.

    Stratifying keeps every fold's class mix comparable, which matters
    here because the deep levels have many small classes -- an unstratified
    split can leave a category entirely absent from a training fold and
    guarantee its rows are wrong.
    """
    n = len(labels)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if not stratify:
        rng = np.random.default_rng(random_state)
        folds = np.arange(n) % n_folds
        rng.shuffle(folds)
        return folds

    from sklearn.model_selection import StratifiedKFold

    y = labels.astype(str).to_numpy()
    counts = pd.Series(y).value_counts()
    if counts.min() < n_folds:
        raise ValueError(
            f"{int((counts < n_folds).sum())} categories have fewer than "
            f"{n_folds} members, so a stratified {n_folds}-fold split is "
            "impossible. Raise SplitConfig.min_path_count to at least "
            f"{n_folds}, lower n_folds, or pass stratify=False."
        )
    folds = np.empty(n, dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for k, (_, test_idx) in enumerate(skf.split(np.zeros(n), y)):
        folds[test_idx] = k
    return folds


# ---------------------------------------------------------------------------
# The expensive part
# ---------------------------------------------------------------------------

def out_of_fold_predictions(
    cfg: PipelineConfig,
    df: Optional[pd.DataFrame] = None,
    n_folds: int = 5,
    cleaning_cfg: Optional[CleaningConfig] = None,
    random_state: int = 42,
    spark=None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Prefix probabilities for every usable row, from a model that never
    saw it.

    Applies the same cleaning and filtering as run(), then folds. Only one
    model is held in memory at a time.

    Returns (prefix, actual, folds, frame):
        prefix  per-depth prefix probabilities and labels, all rows
        actual  the true levels, aligned to prefix
        folds   fold id per row, for reuse when cross-fitting thresholds
        frame   the filtered source rows, aligned to prefix, so ids and
                descriptions can be joined onto per-item output
    """
    levels = cfg.active_levels()
    if df is None:
        df = data_mod.load_data(cfg.data, spark=spark)
    if cfg.clean_text:
        df = clean_dataframe(df, cfg.data.text_col, cleaning_cfg)
        print("[cv] applied text cleaning")

    df = data_mod.drop_null_rows(
        df, cfg.data.text_col, levels, cfg.data.drop_null_text, cfg.data.drop_null_labels
    )
    df = data_mod.drop_rare_paths(df, levels, cfg.split.min_path_count)
    df = df.reset_index(drop=True)

    folds = make_folds(df[levels[-1]], n_folds, random_state, cfg.split.stratify)
    if verbose:
        print(f"[cv] {len(df):,} usable rows, {n_folds} folds "
              f"({np.bincount(folds).min():,}-{np.bincount(folds).max():,} rows each)")

    pieces = []
    for k in range(n_folds):
        train_mask, score_mask = folds != k, folds == k
        if verbose:
            print(f"[cv] fold {k + 1}/{n_folds}: training on "
                  f"{int(train_mask.sum()):,}, scoring {int(score_mask.sum()):,}")
        clf = HierarchicalPathClassifier(level_columns=levels, config=cfg.model)
        clf.fit(df.loc[train_mask, cfg.data.text_col], df.loc[train_mask, levels])

        part = clf.predict_prefix_probabilities(df.loc[score_mask, cfg.data.text_col])
        part.index = np.flatnonzero(score_mask)
        pieces.append(part)

        del clf, part
        gc.collect()

    prefix = pd.concat(pieces).sort_index().reset_index(drop=True)
    actual = df[levels].astype(str).reset_index(drop=True)
    return prefix, actual, folds, df


# ---------------------------------------------------------------------------
# Option A: thresholds from an accuracy target
# ---------------------------------------------------------------------------

def crossfit_accuracy(
    prefix: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    folds: np.ndarray,
    target_accuracy: float = 0.95,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, float], pd.DataFrame]:
    """Cross-fitted depth decisions and the thresholds to deploy.

    Returns (assigned, thresholds, fold_thresholds):
        assigned         a decision for every row, from thresholds fitted
                         without it -- what the report should be built on
        thresholds       fitted on all rows, for deployment
        fold_thresholds  per-fold values; wide spread at a depth means its
                         cut-off is guesswork, whatever the report says
    """
    pieces, rows = [], []
    for k in sorted(set(folds)):
        fit, apply_to = folds != k, folds == k
        th = tune_thresholds(
            prefix[fit].reset_index(drop=True),
            actual[fit].reset_index(drop=True),
            level_columns,
            target_accuracy=target_accuracy,
        )
        rows.append({"fold": int(k), **{f"depth_{d}": t for d, t in th.items()}})
        part = apply_cascade(prefix[apply_to].reset_index(drop=True), level_columns, th)
        part.index = np.flatnonzero(apply_to)
        pieces.append(part)

    assigned = pd.concat(pieces).sort_index().reset_index(drop=True)
    fold_thresholds = pd.DataFrame(rows)
    thresholds = tune_thresholds(
        prefix, actual, level_columns, target_accuracy=target_accuracy
    )

    if verbose:
        _print_spread("accuracy", fold_thresholds)
        print("[cv] thresholds to deploy:",
              {d: round(t, 4) for d, t in thresholds.items()})
    return assigned, thresholds, fold_thresholds


# ---------------------------------------------------------------------------
# Option B: thresholds that minimise cost
# ---------------------------------------------------------------------------

def crossfit_cost(
    prefix: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    folds: np.ndarray,
    costs: CostModel,
    n_grid: int = 200,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, float], pd.DataFrame]:
    """Cross-fitted depth decisions and the cost-optimal thresholds.

    Same shape as crossfit_accuracy, and the same reason for folding --
    more so here, since searching for a cost minimum on a set of rows
    flatters itself on those rows more than merely meeting a target does.

    No calibration anywhere: the search reads accuracy off the labels, so
    it does not matter that the model's own confidence figures run
    optimistic.

    fold_thresholds carries in-sample and out-of-fold cost per item. The
    gap between them is the optimism the fold structure is there to
    expose; the out-of-fold column is the number to quote.
    """
    pieces, rows = [], []
    for k in sorted(set(folds)):
        fit, apply_to = folds != k, folds == k
        p_fit = prefix[fit].reset_index(drop=True)
        a_fit = actual[fit].reset_index(drop=True)
        p_out = prefix[apply_to].reset_index(drop=True)
        a_out = actual[apply_to].reset_index(drop=True)

        th, _ = cost_optimal_thresholds(
            p_fit, a_fit, level_columns, costs, n_grid=n_grid, verbose=False
        )
        in_s = realised_cost(
            apply_cascade(p_fit, level_columns, th), p_fit, a_fit, level_columns, costs
        )
        part = apply_cascade(p_out, level_columns, th)
        out_s = realised_cost(part, p_out, a_out, level_columns, costs)

        rows.append(
            {
                "fold": int(k),
                **{f"depth_{d}": t for d, t in th.items()},
                "cost_per_item_in_sample": in_s["cost_per_item"],
                "cost_per_item_out_of_fold": out_s["cost_per_item"],
                "optimism": out_s["cost_per_item"] - in_s["cost_per_item"],
            }
        )
        part.index = np.flatnonzero(apply_to)
        pieces.append(part)

    assigned = pd.concat(pieces).sort_index().reset_index(drop=True)
    fold_thresholds = pd.DataFrame(rows)
    thresholds, _ = cost_optimal_thresholds(
        prefix, actual, level_columns, costs, n_grid=n_grid, verbose=False
    )

    if verbose:
        _print_spread("cost", fold_thresholds)
        print(f"[cv] out-of-fold cost per item: "
              f"{fold_thresholds['cost_per_item_out_of_fold'].mean():.4f} "
              f"(in-sample {fold_thresholds['cost_per_item_in_sample'].mean():.4f}, "
              f"optimism {fold_thresholds['optimism'].mean():+.4f})")
        print("[cv] thresholds to deploy:",
              {d: round(t, 4) for d, t in thresholds.items()})
    return assigned, thresholds, fold_thresholds


def _print_spread(label: str, fold_thresholds: pd.DataFrame) -> None:
    cols = [c for c in fold_thresholds.columns if c.startswith("depth_")]
    spread = fold_thresholds[cols].replace(np.inf, np.nan)
    for c in cols:
        col = spread[c].dropna()
        if len(col):
            print(f"[cv] {label} {c}: {col.min():.4f}-{col.max():.4f} across folds")
        else:
            print(f"[cv] {label} {c}: never reachable")


# ---------------------------------------------------------------------------
# Savings
# ---------------------------------------------------------------------------

def savings_summary(
    assigned: pd.DataFrame,
    prefix: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    costs: CostModel,
) -> pd.DataFrame:
    """What the policy costs, against doing everything by hand.

    The baseline is depth 0 for every row: today's process. Savings are
    reported per item and for the whole file, because the per-item figure
    looks trivially small and the total usually does not.
    """
    n = len(assigned)
    policy = realised_cost(assigned, prefix, actual, level_columns, costs)
    manual = costs.labour_cost(0) * n

    rows = [
        {"measure": "all manual (today)", "total": manual, "per_item": manual / n},
        {"measure": "with the model", "total": policy["total_cost"],
         "per_item": policy["cost_per_item"]},
        {"measure": "  of which staff time", "total": policy["labour_cost"],
         "per_item": policy["labour_cost"] / n},
        {"measure": "  of which errors", "total": policy["error_cost"],
         "per_item": policy["error_cost"] / n},
        {"measure": "saving", "total": manual - policy["total_cost"],
         "per_item": (manual - policy["total_cost"]) / n},
    ]
    out = pd.DataFrame(rows)
    out.attrs["saving_pct"] = (
        float((manual - policy["total_cost"]) / manual) if manual else np.nan
    )
    out.attrs["n_items"] = n
    out.attrs["n_errors"] = policy["n_errors"]
    out.attrs["mean_depth"] = policy["mean_depth"]
    return out
