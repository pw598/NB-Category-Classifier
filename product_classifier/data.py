"""Loading, filtering, and splitting the golden set."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig, SplitConfig


def load_data(cfg: DataConfig, spark=None, require_labels: bool = True) -> pd.DataFrame:
    """Read a product file as all-string columns.

    Everything is read as str so that IDs like '00123' survive and so that
    pandas never guesses a numeric dtype for a label column.

    require_labels=False skips the check for the hierarchy columns, which
    is what scoring needs: unlabelled records are the whole point there,
    and demanding the columns the model is meant to create would refuse
    exactly the input it exists to handle.
    """
    if cfg.file_format == "csv":
        df = pd.read_csv(cfg.path, delimiter=cfg.delimiter, dtype="str")
    elif cfg.file_format == "excel":
        df = pd.read_excel(cfg.path, dtype="str")
    elif cfg.file_format in ("delta", "spark_table"):
        if spark is None:
            raise ValueError(
                f"file_format={cfg.file_format!r} requires a spark session; "
                "pass spark=spark from the notebook."
            )
        sdf = (
            spark.read.format("delta").load(cfg.path)
            if cfg.file_format == "delta"
            else spark.table(cfg.path)
        )
        df = sdf.toPandas().astype("str")
    else:
        raise ValueError(f"Unsupported file_format: {cfg.file_format!r}")

    validate_columns(df, cfg, require_labels=require_labels)
    return df


def validate_columns(
    df: pd.DataFrame, cfg: DataConfig, require_labels: bool = True
) -> None:
    """Fail early and loudly on a schema mismatch.

    require_labels=False checks only the description and id columns --
    everything needed to *make* a prediction, as opposed to everything
    needed to train on one.
    """
    expected = [cfg.text_col]
    if require_labels:
        expected += list(cfg.level_columns)
    if cfg.id_col:
        expected.append(cfg.id_col)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columns missing from the input data: {missing}. "
            f"Found: {list(df.columns)}"
        )


def drop_null_rows(
    df: pd.DataFrame,
    text_col: str,
    level_columns: List[str],
    drop_null_text: bool = True,
    drop_null_labels: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop rows that cannot be used for training or scoring."""
    n_before = len(df)
    if drop_null_text:
        df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]
    if drop_null_labels:
        df = df.dropna(subset=level_columns)
    if verbose and len(df) < n_before:
        print(f"[data] dropped {n_before - len(df):,} rows with null text/labels")
    return df.reset_index(drop=True)


def drop_rare_paths(
    df: pd.DataFrame,
    level_columns: List[str],
    min_count: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove rows whose full hierarchy path occurs fewer than min_count times.

    Rare paths add classes the model has almost no evidence for and break
    stratified splitting (a class needs >= 2 members to appear in both
    halves).
    """
    if min_count <= 1:
        return df.reset_index(drop=True)

    path_counts = df.groupby(level_columns, dropna=False)[level_columns[0]].transform("size")
    keep = path_counts >= min_count
    if verbose and (~keep).any():
        n_paths_before = df[level_columns].drop_duplicates().shape[0]
        kept = df[keep]
        n_paths_after = kept[level_columns].drop_duplicates().shape[0]
        print(
            f"[data] dropped {int((~keep).sum()):,} rows in paths seen "
            f"< {min_count} times ({n_paths_before:,} -> {n_paths_after:,} distinct paths)"
        )
    return df[keep].reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    level_columns: List[str],
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the deepest active level.

    Stratifying on the deepest level implicitly stratifies the shallower
    ones, since the hierarchy is a tree.
    """
    stratify = df[level_columns[-1]] if cfg.stratify else None
    if stratify is not None:
        too_small = stratify.value_counts()
        too_small = too_small[too_small < 2]
        if len(too_small):
            raise ValueError(
                f"{len(too_small)} classes at '{level_columns[-1]}' have a single "
                "member, so a stratified split is impossible. Raise "
                "SplitConfig.min_path_count (>= 2) or set stratify=False."
            )
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )
    print(f"[data] train={len(train_df):,} rows, test={len(test_df):,} rows")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    split_cfg: SplitConfig,
    level_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Null-drop, rare-path filter, and split, in that order."""
    levels = level_columns or data_cfg.level_columns
    df = drop_null_rows(
        df,
        data_cfg.text_col,
        levels,
        data_cfg.drop_null_text,
        data_cfg.drop_null_labels,
    )
    df = drop_rare_paths(df, levels, split_cfg.min_path_count)
    return split_data(df, levels, split_cfg)
