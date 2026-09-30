"""Grid search over model and vectorizer settings, scored by k-fold CV.

Specify only the parameters you want to vary. Anything you leave out keeps
whatever value the config already has, so a search over one parameter is
one line and adding a second does not require restating the first.

    GRID = {
        "model.vectorizer.max_features": [20_000, 40_000, 80_000],
        "model.nb.alpha": [0.001, 0.01, 0.1],
    }
    results = grid_search(cfg, df, GRID, n_folds=3)

Parameters are addressed by their dotted path from the config root, which
is the same path you would type to set one by hand -- so
`cfg.model.nb.alpha` is `"model.nb.alpha"`. Call `searchable_parameters()`
for the full list with current values.

Scoring is always at full depth. The point of a search is to find settings
that predict the whole path well; depth is a separate decision, made
afterwards by the cascade, and mixing the two makes results incomparable
across runs.

Every candidate is scored by k-fold cross-validation on the same folds, so
differences between candidates are not differences in which rows they
happened to get. Runtime is (candidates x folds) model fits, which grows
quickly: use `sample_frac` for a first pass over a wide grid, then re-run
the shortlist on everything.
"""

from __future__ import annotations

import copy
import gc
import itertools
import json
import os
import time
from datetime import datetime, timezone
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data as data_mod
from .config import PipelineConfig
from .evaluation import path_accuracy, prefix_accuracy
from .model import HierarchicalPathClassifier

# Paths worth searching. Everything else in the config is plumbing, output
# formatting, or something that changes what is being measured rather than
# how well it is measured.
SUGGESTED_PATHS = [
    "model.vectorizer.kind",
    "model.vectorizer.max_features",
    "model.vectorizer.ngram_range",
    "model.vectorizer.min_df",
    "model.vectorizer.max_df",
    "model.vectorizer.stop_words",
    "model.vectorizer.sublinear_tf",
    "model.vectorizer.binary",
    "model.vectorizer.norm",
    "model.vectorizer.use_idf",
    "model.vectorizer.unordered_bigrams",
    "model.nb.alpha",
    "model.nb.fit_prior",
    "model.prediction_mode",
    "split.min_path_count",
]

# What each one does, and values worth trying. Kept beside the paths so the
# reference table can be read without opening config.py.
PARAMETER_NOTES = {
    "model.vectorizer.kind": (
        "tfidf weights rare words up; count uses raw frequencies",
        ["tfidf", "count"]),
    "model.vectorizer.max_features": (
        "vocabulary cap. Bigger reads more descriptions but costs memory "
        "linearly, and min_df often binds first",
        [20_000, 50_000, 100_000, None]),
    "model.vectorizer.ngram_range": (
        "(1,1) single words; (1,2) adds adjacent pairs, which catches "
        "'ball valve' as a unit",
        [(1, 1), (1, 2), (1, 3)]),
    "model.vectorizer.min_df": (
        "drop words appearing in fewer than this many descriptions. 1 keeps "
        "every typo and part number",
        [1, 2, 5]),
    "model.vectorizer.max_df": (
        "drop words appearing in more than this share -- they carry no "
        "discriminating information",
        [0.9, 0.95, 1.0]),
    "model.vectorizer.stop_words": (
        "'english' removes common function words; None keeps them",
        ["english", None]),
    "model.vectorizer.sublinear_tf": (
        "dampen repeated words with 1+log(tf); usually helps short text",
        [True, False]),
    "model.vectorizer.binary": (
        "presence rather than count. count vectorizer only",
        [True, False]),
    "model.vectorizer.norm": (
        "row normalisation, so long descriptions do not dominate",
        ["l2", "l1", None]),
    "model.vectorizer.unordered_bigrams": (
        "fold each bigram to its sorted form, so 'ball valve' and 'valve "
        "ball' are one feature. Needs 2 in ngram_range",
        [False, True]),
    "model.vectorizer.use_idf": (
        "weight by inverse document frequency. tfidf only",
        [True, False]),
    "model.nb.alpha": (
        "additive smoothing. Smaller trusts the training counts more and "
        "sharpens the probabilities",
        [0.001, 0.01, 0.1, 1.0]),
    "model.nb.fit_prior": (
        "learn class frequencies from the data; False assumes every "
        "category equally likely",
        [True, False]),
    "model.prediction_mode": (
        "joint_path multiplies the per-level marginals; conditional_path "
        "uses the chain rule, one model per parent; per_level takes each "
        "level's own argmax and may be internally inconsistent",
        ["joint_path", "per_level", "conditional_path"]),
    "split.min_path_count": (
        "RARE-PATH THRESHOLD: drop categories with fewer than this many "
        "products. Changes which rows exist, so candidates are scored on "
        "different data -- read n_rows alongside the accuracy",
        [10, 20, 50, 100]),
}


# ---------------------------------------------------------------------------
# Addressing config values by path
# ---------------------------------------------------------------------------

def get_by_path(cfg: Any, path: str) -> Any:
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_by_path(cfg: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f"{path!r} does not exist on the config.")
    setattr(obj, parts[-1], value)


def searchable_parameters(cfg: Optional[PipelineConfig] = None) -> pd.DataFrame:
    """Every searchable path, with its default, its current value and notes.

    The default column comes from a fresh PipelineConfig, so `changed`
    shows at a glance what the config in hand has already moved away from
    -- which is usually the first question when a result looks unexpected.

    Pass no config to see the defaults alone.
    """
    reference = PipelineConfig()
    cfg = cfg if cfg is not None else reference

    rows = []
    for path in SUGGESTED_PATHS:
        try:
            current = get_by_path(cfg, path)
            default = get_by_path(reference, path)
        except AttributeError:
            continue
        note, examples = PARAMETER_NOTES.get(path, ("", []))
        rows.append({
            "parameter": path,
            "default": repr(default),
            "current": repr(current),
            "changed": current != default,
            "example_values": ", ".join(repr(v) for v in examples),
            "what_it_does": note,
        })
    return pd.DataFrame(rows)


def print_parameter_reference(cfg: Optional[PipelineConfig] = None) -> None:
    """The same information as readable text, for wide terminals and
    notebooks where a DataFrame truncates the notes column."""
    table = searchable_parameters(cfg)
    for _, r in table.iterrows():
        flag = "  <- changed" if r["changed"] else ""
        print(f"\n{r['parameter']}{flag}")
        print(f"    default : {r['default']}")
        if r["changed"]:
            print(f"    current : {r['current']}")
        if r["what_it_does"]:
            print(f"    {r['what_it_does']}")
        if r["example_values"]:
            print(f"    try     : {r['example_values']}")


def validate_grid(cfg: PipelineConfig, grid: Dict[str, Sequence[Any]]) -> None:
    """Fail early on a bad path or an empty value list."""
    for path, values in grid.items():
        try:
            get_by_path(cfg, path)
        except AttributeError:
            near = [p for p in SUGGESTED_PATHS if path.split(".")[-1] in p]
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise AttributeError(f"{path!r} is not a config path.{hint}") from None
        if not len(list(values)):
            raise ValueError(f"{path!r} has no values to try.")
    if "model.n_levels" in grid:
        raise ValueError(
            "Do not search model.n_levels. Grid search scores at full depth "
            "by design; how deep to predict per product is decided afterwards "
            "by the cascade."
        )


def expand_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Every combination, as a list of {path: value} dicts.

    An empty grid gives one candidate with no overrides -- the config as
    it stands. That is a useful baseline rather than an error.
    """
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    levels: List[str],
    n_folds: int,
    sample_frac: Optional[float],
    random_state: int,
    verbose: bool,
    min_path_count: Optional[int] = None,
) -> pd.DataFrame:
    """Drop nulls, optionally sample, then drop rare paths.

    min_path_count defaults to the floor the folds require -- every class
    needs at least n_folds members for a stratified split to be possible.
    A searched value below that floor is raised to it, since the
    alternative is a crash rather than a result.
    """
    df = data_mod.drop_null_rows(
        df, cfg.data.text_col, levels,
        cfg.data.drop_null_text, cfg.data.drop_null_labels, verbose=verbose,
    )
    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1].")
        df = (
            df.groupby(levels[-1], group_keys=False, observed=True)
            .apply(lambda g: g.sample(frac=sample_frac, random_state=random_state))
            .reset_index(drop=True)
        )
        if verbose:
            print(f"[grid] sampled {sample_frac:.0%} -> {len(df):,} rows")

    floor = max(n_folds, 2)
    threshold = floor if min_path_count is None else max(int(min_path_count), floor)
    df = data_mod.drop_rare_paths(df, levels, threshold, verbose=verbose)
    out = df.reset_index(drop=True)
    out.attrs["min_path_count"] = threshold
    return out


def _folds(labels: pd.Series, n_folds: int, random_state: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    y = labels.astype(str).to_numpy()
    folds = np.empty(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for k, (_, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[test_idx] = k
    return folds


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def grid_search(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    grid: Dict[str, Sequence[Any]],
    n_folds: int = 3,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score every combination by k-fold CV, at full depth.

    Returns one row per candidate, sorted best first:

        path_accuracy       mean across folds -- the ranking column
        path_accuracy_std   spread across folds; see the note below
        correct_through_*   accuracy of the path truncated at each level
        fit_seconds         mean seconds per fold, so you can see what a
                            setting costs as well as what it buys

    Read the standard deviation before believing a winner. If the gap
    between first and second place is smaller than the fold-to-fold
    spread, the ordering is noise and either candidate would do -- in
    which case prefer the cheaper or simpler one.

    A note on searching split.min_path_count: it changes which rows are in
    the data, so candidates with different values are scored on different
    row sets and their accuracies are not strictly comparable. It is
    allowed because it is genuinely worth tuning, but treat differences it
    produces with more suspicion than the rest.
    """
    validate_grid(cfg, grid)

    base = copy.deepcopy(cfg)
    base.model.n_levels = len(base.data.level_columns)   # always full depth
    levels = base.active_levels()

    candidates = expand_grid(grid)

    # split.min_path_count changes which rows exist, so it cannot be
    # applied once up front -- the data has to be re-filtered per
    # candidate, and the folds recomputed with it.
    varies_rows = "split.min_path_count" in grid

    frame = fold_ids = None
    if not varies_rows:
        frame = _prepare(base, df, levels, n_folds, sample_frac,
                         random_state, verbose)
        fold_ids = _folds(frame[levels[-1]], n_folds, random_state)

    if verbose:
        where = f"over {len(frame):,} rows" if frame is not None else "re-filtered per candidate"
        print(f"[grid] {len(candidates)} candidate(s) x {n_folds} folds = "
              f"{len(candidates) * n_folds} fits {where}, "
              f"scoring all {len(levels)} levels")
        if varies_rows:
            print("[grid] searching split.min_path_count: each candidate is "
                  "scored on a different row set, so accuracies are not "
                  "strictly comparable. Compare n_rows alongside.")

    rows = []
    for c_i, overrides in enumerate(candidates, 1):
        run_cfg = copy.deepcopy(base)
        for path, value in overrides.items():
            set_by_path(run_cfg, path, value)

        if varies_rows:
            frame = _prepare(run_cfg, df, levels, n_folds, sample_frac,
                             random_state, False,
                             min_path_count=run_cfg.split.min_path_count)
            fold_ids = _folds(frame[levels[-1]], n_folds, random_state)

        fold_scores, fold_prefix, seconds = [], [], []
        failed = None
        for k in range(n_folds):
            train, test = fold_ids != k, fold_ids == k
            t0 = time.time()
            try:
                clf = HierarchicalPathClassifier(
                    level_columns=levels, config=run_cfg.model
                )
                clf.fit(frame.loc[train, run_cfg.data.text_col],
                        frame.loc[train, levels])
                pred = clf.predict(frame.loc[test, run_cfg.data.text_col])
            except Exception as exc:               # a bad combination is data
                failed = f"{type(exc).__name__}: {exc}"
                break
            truth = frame.loc[test, levels].astype(str).reset_index(drop=True)
            fold_scores.append(path_accuracy(pred, truth, levels))
            fold_prefix.append(prefix_accuracy(pred, truth, levels))
            seconds.append(time.time() - t0)
            del clf, pred
            gc.collect()

        row = {**{p: _display(v) for p, v in overrides.items()}}
        row["n_rows"] = len(frame)
        row["n_paths"] = int(frame[levels].drop_duplicates().shape[0])
        row["min_path_count_applied"] = int(frame.attrs.get("min_path_count", 0))
        if failed:
            row.update({"path_accuracy": np.nan, "path_accuracy_std": np.nan,
                        "fit_seconds": np.nan, "error": failed})
        else:
            row["path_accuracy"] = float(np.mean(fold_scores))
            row["path_accuracy_std"] = float(np.std(fold_scores))
            for key in fold_prefix[0]:
                row[key] = float(np.mean([f[key] for f in fold_prefix]))
            row["fit_seconds"] = float(np.mean(seconds))
            row["error"] = ""
        rows.append(row)

        if verbose:
            got = "FAILED" if failed else f"{row['path_accuracy']:.4f}"
            print(f"[grid] {c_i}/{len(candidates)}  {overrides}  -> {got}")

    results = pd.DataFrame(rows).sort_values(
        "path_accuracy", ascending=False, na_position="last"
    ).reset_index(drop=True)
    results.attrs["grid"] = grid
    results.attrs["n_folds"] = n_folds
    results.attrs["n_rows"] = len(frame)
    if not results["error"].astype(bool).any():
        results = results.drop(columns=["error"])
    return results


def _display(value: Any) -> Any:
    """Tuples and lists become strings so the results table stays readable
    and groupable."""
    return str(value) if isinstance(value, (tuple, list)) else value


# ---------------------------------------------------------------------------
# Using the result
# ---------------------------------------------------------------------------

def best_params(results: pd.DataFrame) -> Dict[str, Any]:
    """The winning combination, as the original {path: value} dict.

    Values come back from the grid rather than the results table, so
    tuples arrive as tuples rather than the strings used for display.
    """
    grid = results.attrs.get("grid", {})
    if not grid:
        return {}
    top = results.iloc[0]
    out = {}
    for path, values in grid.items():
        shown = top[path]
        match = [v for v in values if _display(v) == shown]
        out[path] = match[0] if match else shown
    return out


def apply_params(cfg: PipelineConfig, params: Dict[str, Any]) -> PipelineConfig:
    """A copy of cfg with these parameters set. Leaves the original alone."""
    out = copy.deepcopy(cfg)
    for path, value in params.items():
        set_by_path(out, path, value)
    return out


def save_params(
    results: pd.DataFrame,
    path: str,
    notes: str = "",
) -> Dict[str, Any]:
    """Write the winning parameters to JSON, with how they were found.

    The record carries more than the parameters because a bare dict of
    settings invites the obvious question -- searched over what, on how
    many folds, on how much of the data -- and a file that cannot answer
    it gets distrusted and re-derived.

    Tuples do not survive JSON, so each value's type is recorded
    alongside it and restored by load_params. Without that,
    ngram_range comes back as [1, 2] and sklearn is handed a list where
    it expects a pair.
    """
    params = best_params(results)
    top = results.iloc[0] if len(results) else None

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in params.items()},
        "param_types": {k: type(v).__name__ for k, v in params.items()},
        "grid": {k: [list(x) if isinstance(x, tuple) else x for x in v]
                 for k, v in (results.attrs.get("grid") or {}).items()},
        "n_folds": results.attrs.get("n_folds"),
        "n_rows_searched": results.attrs.get("n_rows"),
        "n_candidates": int(len(results)),
        "path_accuracy": float(top["path_accuracy"]) if top is not None else None,
        "path_accuracy_std": float(top["path_accuracy_std"]) if top is not None else None,
        "notes": notes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record


def load_params(path: str) -> Dict[str, Any]:
    """Read a saved record back, restoring tuple-valued parameters."""
    with open(path, encoding="utf-8") as fh:
        record = json.load(fh)

    types = record.get("param_types", {})
    restored = {}
    for key, value in record.get("params", {}).items():
        if types.get(key) == "tuple" and isinstance(value, list):
            value = tuple(value)
        restored[key] = value
    record["params"] = restored
    return record


def apply_saved_params(cfg: PipelineConfig, path: str, verbose: bool = True):
    """A copy of cfg with the parameters from a saved record applied.

    Returns (cfg, record) so the caller can show where the settings came
    from rather than just asserting them.
    """
    record = load_params(path)
    out = apply_params(cfg, record["params"])
    if verbose:
        acc = record.get("path_accuracy")
        print(f"applied {len(record['params'])} parameter(s) from {path}")
        print(f"  searched {record.get('n_candidates')} candidates on "
              f"{record.get('n_folds')} folds over "
              f"{record.get('n_rows_searched'):,} rows"
              if record.get("n_rows_searched") else "")
        if acc is not None:
            print(f"  best full-path accuracy: {acc:.4f} "
                  f"(+/- {record.get('path_accuracy_std', 0):.4f} across folds)")
        for k, v in record["params"].items():
            print(f"    {k} = {v!r}")
    return out, record


def summarise_by(results: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Marginal effect of one parameter, averaged over the others.

    More trustworthy than reading the single best row when the grid is
    large: it answers "does this parameter matter at all?", which is
    usually the question, and is far less prone to being led by one lucky
    combination.
    """
    if parameter not in results.columns:
        raise KeyError(f"{parameter!r} is not in the results.")
    return (
        results.groupby(parameter, dropna=False)
        .agg(mean_accuracy=("path_accuracy", "mean"),
             best_accuracy=("path_accuracy", "max"),
             n_candidates=("path_accuracy", "size"),
             mean_fit_seconds=("fit_seconds", "mean"))
        .sort_values("mean_accuracy", ascending=False)
        .reset_index()
    )
