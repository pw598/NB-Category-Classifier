"""Loading, filtering, cleaning and saving datasets.

Raw data comes either from a named SQL query in data/queries/<name>.sql
(run with spark.sql on Databricks) or from a delimited text file supplied by
the user. Everything is read as strings.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import __version__, paths
from .cleaning import Cleaner, ngram_stats

DEFAULT_DELIMITER = "|"


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------

def read_query(name: str) -> str:
    """SQL text of data/queries/<name>.sql (or .txt)."""
    for ext in (".sql", ".txt", ""):
        p = paths.QUERY_DIR / f"{name}{ext}"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    available = sorted(p.stem for p in paths.QUERY_DIR.glob("*.sql"))
    raise FileNotFoundError(f"No query named {name!r} in {paths.QUERY_DIR}. Available: {available}")


def _to_str_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every column as Python str, nulls kept as None (not the string 'None')."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        col = df[c]
        out[c] = col.astype(object).where(col.notna(), None).map(
            lambda v: v if v is None or isinstance(v, str) else str(v)
        )
    return out


def run_query(name: str, spark=None) -> pd.DataFrame:
    """Run a named query with Spark and return a pandas DataFrame of strings."""
    sql = read_query(name)
    if spark is None:
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        except Exception as exc:  # pragma: no cover - only on non-Databricks machines
            raise RuntimeError(
                "No Spark session available to run the SQL query. On Databricks this is "
                "automatic; elsewhere provide RAW_DATA_FILE instead."
            ) from exc
    sdf = spark.sql(sql)
    # Cast in Spark: an integer column with nulls would otherwise reach pandas as
    # float64 and become "12345.0", breaking IDs, fold hashes and blacklist matches.
    from pyspark.sql import functions as F

    sdf = sdf.select([F.col(f"`{c}`").cast("string").alias(c) for c in sdf.columns])
    return _to_str_frame(sdf.toPandas())


def read_file(path, delimiter: str = DEFAULT_DELIMITER, quoted: bool = True) -> pd.DataFrame:
    """A delimited .txt/.csv file, every column as str.

    quoted=False treats double quotes as ordinary characters (for raw extracts
    where a description may start with an inch mark, e.g. '"T" HANDLE').
    """
    path = paths.resolve(path)
    extra = {} if quoted else {"quoting": csv.QUOTE_NONE}
    return pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False, na_values=NA_VALUES, **extra)


# Strings treated as null when reading files. Deliberately short: pandas'
# default list includes 'NA' and 'N/A', which can be legitimate values.
NA_VALUES = ["", "NULL", "null", "None", "nan", "NaN", "<NA>"]


def load_raw(
    raw_data_file=None,
    sql_query: Optional[str] = None,
    delimiter: str = DEFAULT_DELIMITER,
    spark=None,
    quoted: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Raw data plus a description of where it came from.

    RAW_DATA_FILE supersedes SQL_QUERY when both are given.
    """
    if raw_data_file is not None:
        df = read_file(raw_data_file, delimiter=delimiter, quoted=quoted)
        source = {"type": "file", "path": str(paths.resolve(raw_data_file)), "delimiter": delimiter,
                  "quoted": quoted}
    elif sql_query is not None:
        df = run_query(sql_query, spark=spark)
        sql = read_query(sql_query)
        source = {
            "type": "sql",
            "query_name": sql_query,
            "query_sha256": hashlib.sha256(sql.encode()).hexdigest()[:16],
            "query_text": sql,
        }
    else:
        raise ValueError("Set RAW_DATA_FILE or SQL_QUERY")
    return df.reset_index(drop=True), source


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def read_blacklist(path, has_header: bool = True) -> List[str]:
    """One-column .txt/.csv of item identifiers to exclude."""
    path = paths.resolve(path)
    with open(path, "r", encoding="utf-8-sig") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    if has_header and lines:
        lines = lines[1:]
    values = set()
    for ln in lines:
        v = ln.split(",")[0] if path.suffix.lower() == ".csv" else ln
        v = v.strip().strip('"').strip()
        if v:
            values.add(v)
    return sorted(values)


def apply_blacklist(df: pd.DataFrame, items: Sequence[str], column: str) -> Tuple[pd.DataFrame, int]:
    if column not in df.columns:
        raise KeyError(f"BLACKLIST_COL {column!r} not in data columns {list(df.columns)}")
    mask = df[column].astype(str).str.strip().isin(set(items))
    return df.loc[~mask].reset_index(drop=True), int(mask.sum())


def drop_missing_labels(df: pd.DataFrame, level_columns: Sequence[str]) -> Tuple[pd.DataFrame, int]:
    labels = df[list(level_columns)]
    bad = labels.isna().any(axis=1) | labels.apply(lambda s: s.astype(str).str.strip() == "").any(axis=1)
    return df.loc[~bad].reset_index(drop=True), int(bad.sum())


def filter_rare_paths(
    df: pd.DataFrame, level_columns: Sequence[str], threshold: int
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop rows whose full category path occurs fewer than `threshold` times."""
    cols = list(level_columns)
    sizes = df.groupby(cols, dropna=False)[cols[0]].transform("size")
    keep = sizes >= threshold
    before_paths = df[cols].drop_duplicates().shape[0]
    out = df.loc[keep].reset_index(drop=True)
    info = {
        "rare_path_threshold": int(threshold),
        "rows_before": int(len(df)),
        "rows_dropped": int((~keep).sum()),
        "rows_after": int(len(out)),
        "paths_before": int(before_paths),
        "paths_after": int(out[cols].drop_duplicates().shape[0]),
    }
    return out, info


def duplicate_count(df: pd.DataFrame, column: Optional[str]) -> int:
    if not column or column not in df.columns:
        return 0
    return int(df[column].duplicated(keep=False).sum())


# ---------------------------------------------------------------------------
# One-call preparation (used by notebooks 02-04)
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    df: pd.DataFrame                 # rows kept, with raw + cleaned text
    cleaner: Cleaner
    metadata: Dict[str, Any] = field(default_factory=dict)


def clean_dataframe(
    df: pd.DataFrame,
    cleaner: Cleaner,
    text_col: str,
    cleaned_col: str,
    drop_empty: bool = True,
    fit: bool = False,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add `cleaned_col`; optionally drop rows with blank raw or cleaned text.

    fit=True learns the cleaning method's vocabulary from these rows (training
    data); fit=False applies the cleaner's stored vocabulary (new data).
    """
    if text_col not in df.columns:
        raise KeyError(f"TEXT_COL {text_col!r} not in data columns {list(df.columns)}")
    counts: Dict[str, Any] = {"rows_in": int(len(df))}
    out = df.copy()
    if drop_empty:
        blank = out[text_col].isna() | (out[text_col].astype(str).str.strip() == "")
        out = out.loc[~blank].reset_index(drop=True)
        counts["blank_raw_dropped"] = int(blank.sum())
    cleaned, report = cleaner.clean_frame(out, text_col, fit=fit, verbose=verbose)
    out[cleaned_col] = cleaned.values
    counts["cleaning_report"] = report
    if drop_empty:
        empty = out[cleaned_col].str.strip() == ""
        out = out.loc[~empty].reset_index(drop=True)
        counts["emptied_by_cleaning_dropped"] = int(empty.sum())
    counts["rows_out"] = int(len(out))
    return out, counts


def prepare_dataset(
    *,
    raw_data_file=None,
    sql_query: Optional[str] = None,
    delimiter: str = DEFAULT_DELIMITER,
    blacklist_file=None,
    blacklist_col: Optional[str] = None,
    blacklist_has_header: bool = True,
    cleaner: Optional[Cleaner] = None,
    cleaning_method: Optional[str] = None,
    append_fields: Optional[Dict[str, str]] = None,
    append_nulls_as_none: bool = True,
    text_col: str = "FullDesc",
    cleaned_col: str = "CleanDesc",
    level_columns: Optional[Sequence[str]] = None,
    drop_empty: bool = True,
    raw_data_quoted: bool = True,
    spark=None,
    verbose: bool = False,
) -> PreparedData:
    """Pull (or read), blacklist-filter and clean in one call.

    Pass either cleaning_method (+ append_fields) to build a new cleaner, which
    then learns its vocabulary from this data (training), or an existing
    `cleaner` (e.g. the one saved with a model), which is applied as is.
    If level_columns is given, rows with a missing label are dropped too.
    Prints only the headline counts unless verbose=True.
    """
    from .cleaning import build_cleaner

    raw, source = load_raw(raw_data_file, sql_query, delimiter, spark=spark, quoted=raw_data_quoted)
    n_raw = len(raw)

    n_black = 0
    if blacklist_file is not None:
        items = read_blacklist(blacklist_file, has_header=blacklist_has_header)
        raw, n_black = apply_blacklist(raw, items, blacklist_col)

    n_nolabel = 0
    if level_columns:
        missing = [c for c in level_columns if c not in raw.columns]
        if missing:
            raise KeyError(f"Label column(s) {missing} not in data columns {list(raw.columns)}")
        raw, n_nolabel = drop_missing_labels(raw, level_columns)

    fit = cleaner is None
    if cleaner is None:
        if cleaning_method is None:
            raise ValueError("Give either cleaner= or cleaning_method=")
        cleaner = build_cleaner(cleaning_method, append_fields=append_fields,
                                append_nulls_as_none=append_nulls_as_none, verbose=verbose)
    df, counts = clean_dataframe(raw, cleaner, text_col, cleaned_col, drop_empty=drop_empty,
                                 fit=fit, verbose=verbose)
    stats = ngram_stats(df[cleaned_col])

    row_counts = {
        "raw_rows": int(n_raw),
        "blacklisted_dropped": int(n_black),
        "missing_label_dropped": int(n_nolabel),
        "blank_raw_dropped": counts.get("blank_raw_dropped", 0),
        "emptied_by_cleaning_dropped": counts.get("emptied_by_cleaning_dropped", 0),
        "cleaned_rows": int(len(df)),
    }
    row_counts["total_dropped"] = row_counts["raw_rows"] - row_counts["cleaned_rows"]
    meta = {
        "created": paths.timestamp(),
        "package_version": __version__,
        "source": source,
        "blacklist_file": str(paths.resolve(blacklist_file)) if blacklist_file else None,
        "blacklist_col": blacklist_col,
        "text_col": text_col,
        "cleaned_col": cleaned_col,
        "level_columns": list(level_columns) if level_columns else None,
        "cleaning": cleaner.describe(),
        "row_counts": row_counts,
        "ngram_stats": stats,
    }
    print_summary(row_counts, stats)
    return PreparedData(df=df, cleaner=cleaner, metadata=meta)


def blacklist_prepared(prepared: PreparedData, blacklist_file, blacklist_col: str,
                       has_header: bool = True) -> PreparedData:
    """Apply a blacklist to an already-prepared dataset (e.g. one loaded from notebook 01)."""
    if blacklist_file is None:
        return prepared
    items = read_blacklist(blacklist_file, has_header=has_header)
    df, n = apply_blacklist(prepared.df, items, blacklist_col)
    meta = dict(prepared.metadata)
    rc = dict(meta.get("row_counts", {}))
    rc["blacklisted_dropped"] = rc.get("blacklisted_dropped", 0) + n
    rc["cleaned_rows"] = int(len(df))
    rc["total_dropped"] = rc.get("raw_rows", len(df)) - rc["cleaned_rows"]
    meta["row_counts"] = rc
    meta["extra_blacklist_file"] = str(paths.resolve(blacklist_file))
    print(f"blacklist {blacklist_file}: {n:,} more rows removed")
    return PreparedData(df=df, cleaner=prepared.cleaner, metadata=meta)


def print_summary(row_counts: Dict[str, int], stats: Dict[str, int]) -> None:
    print(f"raw data rows      : {row_counts['raw_rows']:>12,}")
    print(f"rows dropped       : {row_counts['total_dropped']:>12,}")
    print(f"cleaned data rows  : {row_counts['cleaned_rows']:>12,}")
    print(f"unique tokens      : {stats['unique_unigrams']:>12,}")
    print(f"unique bigrams     : {stats['unique_bigrams']:>12,}")
    print(f"tokens + bigrams   : {stats['unique_unigrams_plus_bigrams']:>12,}")


# ---------------------------------------------------------------------------
# Saving / loading prepared datasets (data/<ts>/)
# ---------------------------------------------------------------------------

def save_dataset(
    folder: Path,
    ts: str,
    raw: Optional[pd.DataFrame],
    cleaned: pd.DataFrame,
    cleaner: Cleaner,
    metadata: Dict[str, Any],
    delimiter: str = DEFAULT_DELIMITER,
) -> Dict[str, Path]:
    """Write raw_data_<ts>.txt, cleaned_data_<ts>.txt, cleaner.pkl, vocabulary.txt,
    a copy of the cleaning-method folder, and metadata_<ts>.json."""
    import joblib

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    raw_path = folder / f"raw_data_{ts}.txt"
    if raw is not None:
        raw.to_csv(raw_path, sep=delimiter, index=False)
    if raw_path.is_file():  # written here or earlier by the notebook
        out["raw"] = raw_path
    out["cleaned"] = folder / f"cleaned_data_{ts}.txt"
    cleaned.to_csv(out["cleaned"], sep=delimiter, index=False)
    out["cleaner"] = folder / "cleaner.pkl"
    joblib.dump(cleaner, out["cleaner"], compress=3)
    out.update(cleaner.export(folder))  # copy of the method folder + vocabulary.txt
    meta = dict(metadata)
    meta["files"] = {k: v.name for k, v in out.items()}
    meta["delimiter"] = delimiter
    out["metadata"] = folder / f"metadata_{ts}.json"
    write_json(out["metadata"], meta)
    return out


def load_dataset(name: Optional[str] = "latest") -> PreparedData:
    """A dataset written by notebook 01 (folder name, timestamp, path or 'latest')."""
    import joblib

    if name is None or str(name).lower() == "latest":
        folder = latest_dataset_folder()
    else:
        folder = paths.find_folder(name, paths.DATA_DIR, prefix="")  # data/<ts>/
    metas = sorted(folder.glob("metadata_*.json"))
    if not metas:
        raise FileNotFoundError(f"No metadata_*.json in {folder}")
    meta = json.loads(metas[-1].read_text())
    cleaned_file = folder / meta["files"]["cleaned"]
    df = pd.read_csv(cleaned_file, sep=meta.get("delimiter", DEFAULT_DELIMITER), dtype=str,
                     keep_default_na=False, na_values=NA_VALUES)
    df[meta["cleaned_col"]] = df[meta["cleaned_col"]].fillna("")
    cleaner = joblib.load(folder / "cleaner.pkl")
    meta["loaded_from"] = str(folder)
    return PreparedData(df=df, cleaner=cleaner, metadata=meta)


def latest_dataset_folder() -> Path:
    folders = sorted(p for p in paths.DATA_DIR.iterdir()
                     if p.is_dir() and p.name[:8].isdigit()) if paths.DATA_DIR.is_dir() else []
    if not folders:
        raise FileNotFoundError(f"No dataset folders in {paths.DATA_DIR}")
    return folders[-1]


def write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
