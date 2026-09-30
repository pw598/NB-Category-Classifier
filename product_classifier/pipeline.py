"""End-to-end orchestration: config in, fitted model and scores out.

Everything here is glue. If you find yourself adding logic to this module,
it probably belongs in data.py, model.py, or evaluation.py instead.
"""

from __future__ import annotations

import gc
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from . import data as data_mod
from .cleaning.text import CleaningConfig, clean_dataframe
from .config import PipelineConfig
from .evaluation import EvaluationResult, evaluate
from .model import HierarchicalPathClassifier


@dataclass
class RunResult:
    config: PipelineConfig
    classifier: HierarchicalPathClassifier
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    evaluation: EvaluationResult


def run(
    cfg: PipelineConfig,
    df: Optional[pd.DataFrame] = None,
    cleaning_cfg: Optional[CleaningConfig] = None,
    spark=None,
) -> RunResult:
    """Load, clean, split, fit, and score in one call.

    Pass `df` to skip loading (handy when iterating in a notebook and the
    read is slow).
    """
    levels = cfg.active_levels()
    print(f"[run] predicting {len(levels)} level(s): {levels}")
    print(f"[run] prediction mode: {cfg.model.prediction_mode}")

    if df is None:
        df = data_mod.load_data(cfg.data, spark=spark)
    print(f"[run] loaded {len(df):,} rows")

    if cfg.clean_text:
        df = clean_dataframe(df, cfg.data.text_col, cleaning_cfg)
        print("[run] applied text cleaning")

    train_df, test_df = data_mod.prepare(df, cfg.data, cfg.split, levels)

    clf = HierarchicalPathClassifier(level_columns=levels, config=cfg.model)
    clf.fit(train_df[cfg.data.text_col], train_df[levels])

    result = evaluate(
        clf,
        test_df,
        text_col=cfg.data.text_col,
        id_col=cfg.data.id_col,
        bucket_width=cfg.evaluation.probability_bucket_width,
        compare_against_per_level=cfg.evaluation.compare_against_per_level,
    )
    print("[run] evaluation:")
    print(result.summary_text())

    return RunResult(cfg, clf, train_df, test_df, result)


def _write_frame(df: pd.DataFrame, out_dir: str, stem: str, fmt: str) -> str:
    """Write one frame, falling back to csv if the requested engine is absent.

    A missing parquet engine should not destroy the results of a long
    training run, so this warns and degrades instead of raising.
    """
    if fmt == "parquet":
        path = os.path.join(out_dir, f"{stem}.parquet")
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            warnings.warn(
                "parquet requested but no engine is installed (pip install pyarrow); "
                "writing csv instead.",
                RuntimeWarning,
            )
            fmt = "csv"
    if fmt == "excel":
        path = os.path.join(out_dir, f"{stem}.xlsx")
        try:
            df.to_excel(path, index=False)
            return path
        except ImportError:
            warnings.warn(
                "excel requested but openpyxl is not installed; writing csv instead.",
                RuntimeWarning,
            )
            fmt = "csv"
    path = os.path.join(out_dir, f"{stem}.csv")
    df.to_csv(path, index=False)
    return path


def save_outputs(result: RunResult, prefix: str = "run") -> dict:
    """Write the comparison, bucket, and confusion tables to the output dir.

    Format comes from EvalConfig.output_format. The bucket and confusion
    tables are always csv — they're small, and a human usually wants to open
    them directly.
    """
    out_dir = result.config.evaluation.output_dir
    fmt = result.config.evaluation.output_format
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    paths["comparison"] = _write_frame(
        result.evaluation.comparison, out_dir, f"{prefix}_comparison", fmt
    )
    paths["buckets"] = _write_frame(
        result.evaluation.bucket_summary, out_dir, f"{prefix}_buckets", "csv"
    )
    if result.evaluation.confusion is not None:
        paths["confusions"] = _write_frame(
            result.evaluation.confusion, out_dir, f"{prefix}_confusions", "csv"
        )

    print(f"[run] wrote {len(paths)} file(s) to {out_dir}")
    return paths


def sweep_depths(
    cfg: PipelineConfig,
    depths: Tuple[int, ...] = (1, 2, 3, 4),
    df: Optional[pd.DataFrame] = None,
    spark=None,
) -> pd.DataFrame:
    """Run the pipeline at several hierarchy depths and tabulate the result.

    Useful for the "how far down can we usefully predict?" question: L1
    accuracy is usually high and L4 much lower, and this shows the drop-off
    in one table.
    """
    if df is None:
        df = data_mod.load_data(cfg.data, spark=spark)

    import copy

    rows = []
    for depth in depths:
        depth_cfg = copy.deepcopy(cfg)
        depth_cfg.model.n_levels = depth
        result = run(depth_cfg, df=df)
        row = {"n_levels": depth, **result.evaluation.metrics}
        rows.append(row)
        # Only the metrics are kept, but the RunResult holds the fitted
        # models plus the train/test frames. Without this it stays live
        # through the whole of the next, deeper run.
        del result
        gc.collect()
    return pd.DataFrame(rows)


def compare_modes(
    cfg: PipelineConfig,
    df: Optional[pd.DataFrame] = None,
    spark=None,
    include_conditional: bool = False,
) -> pd.DataFrame:
    """Joint-path vs. per-level, same data, same split, same models.

    include_conditional adds the chain-rule mode. It needs its own fit --
    the per-parent models are a different set of estimators, not a
    different reading of the same ones -- so the run costs roughly twice
    as long and the comparison is across two fits rather than one. The
    split is seeded identically, so the test rows are the same.
    """
    import copy

    if df is None:
        df = data_mod.load_data(cfg.data, spark=spark)

    joint_cfg = copy.deepcopy(cfg)
    joint_cfg.model.prediction_mode = "joint_path"
    joint_cfg.evaluation.compare_against_per_level = True
    result = run(joint_cfg, df=df)

    m = result.evaluation.metrics
    rows = [
        {"mode": "joint_path", "path_accuracy": m["path_accuracy"]},
        {
            "mode": "per_level",
            "path_accuracy": m.get("per_level_mode_path_accuracy"),
        },
    ]

    if include_conditional:
        del result
        gc.collect()
        cond_cfg = copy.deepcopy(cfg)
        cond_cfg.model.prediction_mode = "conditional_path"
        cond_cfg.evaluation.compare_against_per_level = False
        cond = run(cond_cfg, df=df)
        rows.append({
            "mode": "conditional_path",
            "path_accuracy": cond.evaluation.metrics["path_accuracy"],
        })

    return pd.DataFrame(rows)
