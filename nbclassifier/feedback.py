"""Experiments on training with labels that came from the model's own predictions.

Items predicted by notebook 04 and then written back as categories show up in
the labelled pull (data/queries/labeled_items.sql) like any other item. Training
on them can look like a large gain, but two things distort a naive comparison:

* scoring on them is circular (their labels are the model's own predictions);
* the production model that produced them was trained on ALL organic labels,
  so a student trained on them has indirectly seen any organic holdout.

Everything here is therefore cross-fitted. For each fold k of the ORGANIC
rows, a teacher is trained only on the other folds, the teacher re-labels the
pool (and re-applies the acceptance rule), and the students are scored on fold
k, which no teacher or student ever saw. Production labels of derived rows are
used for diagnostics only.

Terms
    organic   labelled rows whose label did not come from a model prediction
    derived   labelled rows whose label did come from a model prediction
    pool      every item the model predicted in the batches the derived rows
              came from (accepted or not); teachers re-label it in each fold
    ratio     derived rows / organic rows in a training set; the production
              ratio is len(derived) / len(organic)

Three experiments (notebooks 05-07) share the machinery below:

    run_ratio_grid       students at many (organic fraction, ratio) cells; the
                         teacher of each cell is trained on that cell's organic
                         rows, as in production (notebooks 05 and 06)
    run_teacher_quality  teacher error vs teacher training size, on clean
                         folds and transferred to the pool (notebook 07)
    run_noise_dose       organic rows posing as the pool, so the pseudo-label
                         noise is known and can be dialled (notebook 07)

Results are checkpointed in outputs/feedback_<kind>_<ts>/ exactly like
notebook 02: one parquet file per finished cell, plus a copy of the data, so
RESUME_FROM skips finished work and never re-pulls.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import paths
from .config import HyperParams
from .data import NA_VALUES, PreparedData, clean_dataframe, filter_rare_paths, write_json
from .folds import id_hash_folds
from .model import HierarchicalNB
from .predict import correct_flags

ROW = "_row"            # position of an organic row (test rows are matched on it)
TRUE_VALUES = ("1", "true", "t", "y", "yes")
PROD = "Production "    # prefix for production prediction columns copied from notebook 04 files


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def hash_uniform(ids, salt: str) -> np.ndarray:
    """A number in [0, 1) per row, decided by its ID alone: sha1(f"{salt}|{id}").

    Subsets taken as `u < fraction` are nested (a smaller fraction is a subset
    of a larger one) and do not depend on which other rows are present.
    """
    keys = pd.Series(ids, dtype=object).astype(str).str.strip()
    out = np.empty(len(keys), dtype=np.float64)
    for i, key in enumerate(keys):
        digest = hashlib.sha1(f"{salt}|{key}".encode("utf-8")).digest()
        out[i] = int.from_bytes(digest[:8], "big") / 2.0 ** 64
    return out


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _fmt(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".") if x != int(x) else str(int(x))


@dataclass
class Cols:
    id: str
    cleaned: str
    levels: List[str]

    @property
    def train_cols(self) -> List[str]:
        return [self.cleaned] + list(self.levels)


@dataclass
class SelectionRule:
    """How predictions were accepted into the labelled data.

    min_confidence None means every predicted item was accepted; otherwise an
    item is accepted when its L<level> Confidence is at least min_confidence.
    """

    level: int = 4
    min_confidence: Optional[float] = None

    def mask(self, pred: pd.DataFrame) -> np.ndarray:
        if self.min_confidence is None:
            return np.ones(len(pred), dtype=bool)
        return pred[f"L{self.level} Confidence"].to_numpy(dtype=float) >= self.min_confidence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Which labelled rows came from predictions
# ---------------------------------------------------------------------------

def prediction_files(spec) -> List[Path]:
    """Notebook-04 prediction CSVs.

    spec: "all" (every outputs/predictions_*/predictions_*.csv), a folder name
    or timestamp ("predictions_20261016_120000"), a CSV path, or a list of these.
    """
    if spec is None:
        return []
    items = [spec] if isinstance(spec, (str, Path)) else list(spec)
    files: List[Path] = []
    for s in items:
        if str(s).lower() == "all":
            if paths.OUTPUTS_DIR.is_dir():
                files += sorted(paths.OUTPUTS_DIR.glob("predictions_*/predictions_*.csv"))
            continue
        p = paths.resolve(s)
        if p is not None and p.is_file():
            files.append(p)
            continue
        folder = paths.find_folder(str(s), paths.OUTPUTS_DIR, prefix="predictions_")
        files += sorted(folder.glob("predictions_*.csv"))
    seen, out = set(), []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def load_prediction_files(spec, id_col: str = "ID") -> Optional[pd.DataFrame]:
    """All rows of the prediction files, one per ID (a later file wins)."""
    files = prediction_files(spec)
    if not files:
        return None
    frames = []
    for f in files:
        t = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=NA_VALUES)
        if id_col not in t.columns:
            raise KeyError(f"{f} has no {id_col!r} column")
        t["_prediction_file"] = f.name
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    df[id_col] = df[id_col].astype(str).str.strip()
    return df.drop_duplicates(id_col, keep="last").reset_index(drop=True)


@dataclass
class FeedbackData:
    organic: pd.DataFrame        # independently labelled rows, with ROW
    derived: pd.DataFrame        # rows whose recorded label came from a prediction
    pool: pd.DataFrame           # id, cleaned text, 'derived' flag (+ production predictions)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def production_ratio(self) -> float:
        return len(self.derived) / max(len(self.organic), 1)

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        for name in ("organic", "derived", "pool"):
            _atomic_parquet(getattr(self, name), folder / f"{name}.parquet")
        write_json(folder / "info.json", self.info)

    @classmethod
    def load(cls, folder: Path) -> "FeedbackData":
        folder = Path(folder)
        parts = {n: pd.read_parquet(folder / f"{n}.parquet") for n in ("organic", "derived", "pool")}
        return cls(**parts, info=json.loads((folder / "info.json").read_text()))


def _clean_pool_rows(extra: pd.DataFrame, cleaner, text_col: str, cleaned_col: str,
                     notes: List[str]) -> Optional[pd.DataFrame]:
    """Clean predicted-but-not-accepted items with the dataset's own cleaner."""
    missing_append = [c for c in (cleaner.append_fields or {}) if c not in extra.columns]
    if text_col in extra.columns and not missing_append:
        out, _ = clean_dataframe(extra, cleaner, text_col, cleaned_col, drop_empty=True, fit=False)
        notes.append(f"{len(out):,} pool items that were not accepted re-cleaned from {text_col!r}")
        return out
    if cleaned_col in extra.columns:
        why = (f"append-field column(s) {missing_append} missing" if missing_append
               else f"no {text_col!r} column")
        notes.append(f"{why} in the prediction files: re-used their {cleaned_col!r} "
                     "(cleaned with the production model's cleaner)")
        out = extra[extra[cleaned_col].fillna("").str.strip() != ""].copy()
        return out
    notes.append(f"prediction files have neither {text_col!r} nor {cleaned_col!r}: "
                 "pool = derived rows only (set INCLUDE_TEXT = True in notebook 04)")
    return None


def split_feedback(
    prepared: PreparedData,
    *,
    level_columns: Sequence[str],
    text_col: str,
    cleaned_col: str,
    id_col: str = "ID",
    prediction_files="all",
    derived_flag_col: Optional[str] = None,
    pool_source: str = "prediction_files",
) -> FeedbackData:
    """Split labelled rows into organic and derived, and build the pool.

    Derived rows are those flagged in `derived_flag_col` or, without a flag
    column, those whose ID appears in a notebook-04 prediction file.

    pool_source "prediction_files": every predicted item (accepted or not) that
                is not organic. Needs the text in the prediction files.
                "derived": only the derived rows (when you know nothing about
                the items that were predicted but not accepted).
    """
    df = prepared.df.copy()
    levels = list(level_columns)
    if id_col not in df.columns:
        raise KeyError(f"ID column {id_col!r} not in data columns {list(df.columns)}")
    df[id_col] = df[id_col].astype(str).str.strip()
    preds = load_prediction_files(prediction_files, id_col) if prediction_files else None

    if derived_flag_col:
        if derived_flag_col not in df.columns:
            raise KeyError(f"DERIVED_FLAG_COL {derived_flag_col!r} not in data columns")
        is_derived = df[derived_flag_col].astype(str).str.strip().str.lower().isin(TRUE_VALUES)
        how = f"flag column {derived_flag_col!r}"
    elif preds is not None:
        is_derived = df[id_col].isin(set(preds[id_col]))
        how = f"ID in {preds['_prediction_file'].nunique()} prediction file(s)"
    else:
        raise ValueError("No prediction files found and no DERIVED_FLAG_COL: cannot tell which "
                         "labelled rows came from predictions")

    organic = df.loc[~is_derived].reset_index(drop=True)
    organic[ROW] = np.arange(len(organic))
    derived = df.loc[is_derived].reset_index(drop=True)
    if not len(derived):
        raise ValueError(f"No derived rows found ({how}); nothing to study")

    notes: List[str] = []
    pool = derived[[id_col, cleaned_col]].copy()
    pool["derived"] = True
    if pool_source == "prediction_files":
        if preds is None:
            notes.append("no prediction files: pool = derived rows only")
        else:
            labelled_ids = set(df[id_col])
            extra = preds[~preds[id_col].isin(labelled_ids)]
            if len(extra):
                cleaned_extra = _clean_pool_rows(extra, prepared.cleaner, text_col, cleaned_col, notes)
                if cleaned_extra is not None and len(cleaned_extra):
                    e = cleaned_extra[[id_col, cleaned_col]].copy()
                    e["derived"] = False
                    pool = pd.concat([pool, e], ignore_index=True)
    elif pool_source != "derived":
        raise ValueError("pool_source must be 'prediction_files' or 'derived'")

    if preds is not None:
        prod_cols = [c for c in [f"Predicted {lv}" for lv in levels]
                     + [f"L{d} Confidence" for d in range(1, len(levels) + 1)] if c in preds.columns]
        prod = preds[[id_col] + prod_cols].rename(columns={c: PROD + c for c in prod_cols})
        for c in prod_cols:
            if c.endswith("Confidence"):
                prod[PROD + c] = pd.to_numeric(prod[PROD + c], errors="coerce")
        pool = pool.merge(prod, on=id_col, how="left")
        derived = derived.merge(prod, on=id_col, how="left")
    pool = pool.drop_duplicates(id_col).reset_index(drop=True)

    info = {
        "derived_identified_by": how,
        "pool_source": pool_source,
        "n_labelled": int(len(df)),
        "n_organic": int(len(organic)),
        "n_derived": int(len(derived)),
        "n_pool": int(len(pool)),
        "production_ratio": float(len(derived) / max(len(organic), 1)),
        "prediction_files": sorted(preds["_prediction_file"].unique().tolist()) if preds is not None else [],
        "notes": notes,
        "id_col": id_col,
        "cleaned_col": cleaned_col,
        "level_columns": levels,
    }
    print(f"organic rows   : {len(organic):>10,}")
    print(f"derived rows   : {len(derived):>10,}   ({how})")
    print(f"pool items     : {len(pool):>10,}   (source: {pool_source})")
    print(f"production ratio derived/organic = {info['production_ratio']:.3f}")
    for n in notes:
        print("note:", n)
    return FeedbackData(organic, derived, pool, info)


def describe(fd: FeedbackData, level_columns: Sequence[str], rule: SelectionRule) -> pd.DataFrame:
    """Diagnostics of the derived rows against the production predictions.

    - path agreement between each derived row's recorded label and the
      production prediction (1.0 = the labels are pure model output; less =
      some were corrected after prediction)
    - production confidence of accepted vs not-accepted pool items, which
      shows the acceptance threshold actually in use
    """
    rows = []
    levels = list(level_columns)
    d = fd.derived
    if all(f"{PROD}Predicted {lv}" in d.columns for lv in levels):
        have = d[f"{PROD}Predicted {levels[-1]}"].notna()
        ok = np.ones(int(have.sum()), dtype=bool)
        for i, lv in enumerate(levels, start=1):
            ok &= (d.loc[have, lv].astype(str).to_numpy()
                   == d.loc[have, f"{PROD}Predicted {lv}"].astype(str).to_numpy())
            rows.append({"check": f"recorded label = production prediction (through L{i})",
                         "value": float(ok.mean()) if len(ok) else np.nan, "n": int(have.sum())})
    conf_col = f"{PROD}L{rule.level} Confidence"
    p = fd.pool
    if conf_col in p.columns:
        for flag, name in ((True, "accepted (derived)"), (False, "predicted, not accepted")):
            c = p.loc[p["derived"] == flag, conf_col].dropna()
            if len(c):
                rows.append({"check": f"production L{rule.level} confidence, {name}: min",
                             "value": float(c.min()), "n": int(len(c))})
                rows.append({"check": f"production L{rule.level} confidence, {name}: median",
                             "value": float(c.median()), "n": int(len(c))})
                rows.append({"check": f"production L{rule.level} confidence, {name}: max",
                             "value": float(c.max()), "n": int(len(c))})
    if len(p) > len(fd.derived):
        rows.append({"check": "acceptance rate (derived / pool)", "value": len(fd.derived) / len(p),
                     "n": int(len(p))})
    rows.append({"check": "production ratio (derived / organic)", "value": fd.production_ratio,
                 "n": int(len(fd.organic))})
    out = pd.DataFrame(rows)
    if rule.min_confidence is None and (out["check"].str.contains("not accepted").any()):
        print("note: the pool holds predicted items that were NOT accepted, but ACCEPT_MIN_CONFIDENCE is None, "
              "so every teacher prediction will be accepted. If acceptance used a threshold, set it.")
    return out


# ---------------------------------------------------------------------------
# Fitting, scoring, pseudo-labelling
# ---------------------------------------------------------------------------

def fit_model(train: pd.DataFrame, hp: HyperParams, cols: Cols) -> Tuple[HierarchicalNB, Dict[str, int]]:
    """Rare-path filter + HierarchicalNB, exactly as notebook 03 trains."""
    train, rare = filter_rare_paths(train[cols.train_cols], cols.levels, hp.rare_path_threshold)
    if not len(train):
        raise ValueError(f"rare_path_threshold={hp.rare_path_threshold} leaves no training rows")
    clf = HierarchicalNB(cols.levels, hp).fit(train[cols.cleaned], train[cols.levels])
    return clf, rare


def score(clf: HierarchicalNB, test: pd.DataFrame, cols: Cols, batch_size: int = 2000) -> pd.DataFrame:
    """Per test row: ROW, L<d> Correct (path prefix), L<d> Confidence."""
    pred = clf.predict_frame(test[cols.cleaned], batch_size=batch_size)
    for c in cols.levels:
        pred[f"Actual {c}"] = test[c].to_numpy()
    ok = correct_flags(pred, cols.levels)
    out = pd.DataFrame({ROW: test[ROW].to_numpy()})
    for d in range(1, len(cols.levels) + 1):
        out[f"L{d} Correct"] = ok[f"L{d} Correct"].to_numpy()
        out[f"L{d} Confidence"] = pred[f"L{d} Confidence"].to_numpy()
    return out


def pseudo_label(clf: HierarchicalNB, pool: pd.DataFrame, cols: Cols, rule: SelectionRule,
                 salt: str, batch_size: int = 2000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Teacher predictions on the pool, accepted by `rule`, in a fixed hash order.

    Taking the first n rows of the result gives nested subsets as n grows.
    """
    pred = clf.predict_frame(pool[cols.cleaned], batch_size=batch_size)
    out = pd.DataFrame({cols.id: pool[cols.id].to_numpy(), cols.cleaned: pool[cols.cleaned].to_numpy()})
    for c in cols.levels:
        out[c] = pred[f"Predicted {c}"].to_numpy()
    for d in range(1, len(cols.levels) + 1):
        out[f"L{d} Confidence"] = pred[f"L{d} Confidence"].to_numpy()
    sel = rule.mask(pred)
    out["_u"] = hash_uniform(out[cols.id], salt)
    accepted = out.loc[sel].sort_values("_u", kind="mergesort").reset_index(drop=True)
    return accepted, {"n_pool": int(len(out)), "n_accepted": int(sel.sum()),
                      "pool_accept_share": float(sel.mean()) if len(sel) else float("nan")}


def _teacher_summary(t: pd.DataFrame, rule: SelectionRule, n_levels: int) -> Dict[str, float]:
    sel = rule.mask(t)
    out = {"test_accept_share": float(sel.mean())}
    for d in range(1, n_levels + 1):
        ok = t[f"L{d} Correct"].to_numpy()
        out[f"teacher L{d} Accuracy"] = float(ok.mean())
        out[f"teacher L{d} error among accepted"] = float(1 - ok[sel].mean()) if sel.any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# Checkpoint store
# ---------------------------------------------------------------------------

FIXED_CONFIG_KEYS = ("kind", "hyperparameters", "n_folds", "fold_seed", "sample_seed", "rule", "columns")


class Store:
    """outputs/feedback_<kind>_<ts>/: config, a copy of the data, one parquet per finished cell."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.config = json.loads((self.folder / "config.json").read_text())

    @classmethod
    def create(cls, kind: str, config: Dict[str, Any], fd: FeedbackData) -> "Store":
        folder = paths.new_folder(paths.OUTPUTS_DIR, f"feedback_{kind}_")
        fd.save(folder / "dataset")
        write_json(folder / "config.json", {"kind": kind, "created": folder.name[-15:],
                                            **json.loads(json.dumps(config, default=str))})
        print(f"[feedback] created {folder}")
        return cls(folder)

    @classmethod
    def open(cls, name: str, kind: str, config: Dict[str, Any]) -> "Store":
        folder = paths.find_folder(name, paths.OUTPUTS_DIR, prefix=f"feedback_{kind}_")
        st = cls(folder)
        new = {"kind": kind, **json.loads(json.dumps(config, default=str))}
        diff = {k: (st.config.get(k), new.get(k)) for k in FIXED_CONFIG_KEYS
                if k in new and st.config.get(k) != new.get(k)}
        if diff:
            raise ValueError("Settings differ from the run being resumed:\n" +
                             "\n".join(f"  {k}: saved {a!r} / notebook {b!r}" for k, (a, b) in diff.items()))
        st.config.update({k: v for k, v in new.items() if k not in FIXED_CONFIG_KEYS})
        write_json(folder / "config.json", st.config)
        print(f"[feedback] resuming {folder.name}: {len(st.keys())} cell(s) already done")
        return st

    def data(self) -> FeedbackData:
        return FeedbackData.load(self.folder / "dataset")

    def _path(self, key: str) -> Path:
        return self.folder / "results" / f"{key}.parquet"

    def done(self, key: str) -> bool:
        return self._path(key).is_file()

    def save(self, key: str, table: pd.DataFrame, meta: Dict[str, Any]) -> None:
        self._path(key).parent.mkdir(parents=True, exist_ok=True)
        write_json(self._path(key).with_suffix(".json"), {"key": key, **meta})
        _atomic_parquet(table, self._path(key))  # written last: its presence marks the cell done

    def table(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(key))

    def keys(self) -> List[str]:
        d = self.folder / "results"
        return sorted(p.stem for p in d.glob("*.parquet")) if d.is_dir() else []

    def meta(self) -> pd.DataFrame:
        import pyarrow.parquet as pq

        rows = []
        for k in self.keys():
            m = json.loads(self._path(k).with_suffix(".json").read_text())
            names = pq.read_schema(self._path(k)).names
            t = pd.read_parquet(self._path(k), columns=[c for c in names if c.endswith("Correct")])
            for col in t.columns:
                m[col.replace("Correct", "Accuracy")] = float(t[col].mean())
            rows.append(m)
        return pd.DataFrame(rows)


def fold_assignment(organic: pd.DataFrame, id_col: str, n_folds: int, fold_seed: int) -> np.ndarray:
    """Same ID-hash folds as notebook 02 (and the NN repo)."""
    return id_hash_folds(organic[id_col], n_folds, fold_seed)


def _progress(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------
# Experiment 1 and 2: students over (organic fraction, ratio)
# ---------------------------------------------------------------------------

def ratio_key(fold: int, frac: float, ratio: float) -> str:
    return f"f{fold}_g{_fmt(frac)}_r{_fmt(ratio)}"


def run_ratio_grid(store: Store, fd: FeedbackData, hp: HyperParams, cols: Cols, *, n_folds: int,
                   fold_seed: int, folds_to_run: Optional[Sequence[int]], organic_fracs: Sequence[float],
                   ratios: Sequence[float], rule: SelectionRule, sample_seed: int = 0,
                   batch_size: int = 2000) -> pd.DataFrame:
    """For each fold k and organic fraction g:

    1. O_g = a nested hash subsample (share g) of the organic rows outside fold k
    2. teacher = model trained on O_g   (this is also the ratio-0 student)
    3. the teacher re-labels the pool; the acceptance rule is re-applied
    4. for each ratio r: student trained on O_g + the first round(r*|O_g|)
       accepted pool items (fewer if not enough were accepted: 'capped')
    5. every model is scored on fold k

    Returns the table of finished cells (store.meta()).
    """
    org = fd.organic
    folds = fold_assignment(org, cols.id, n_folds, fold_seed)
    run_folds = list(range(n_folds)) if folds_to_run is None else list(folds_to_run)
    ratios = sorted({0.0, *[float(r) for r in ratios]})
    fracs = sorted({float(g) for g in organic_fracs})
    if any(not (0 < g <= 1) for g in fracs):
        raise ValueError("organic fractions must be in (0, 1]")
    total = len(run_folds) * len(fracs) * len(ratios)
    done = sum(store.done(ratio_key(k, g, r)) for k in run_folds for g in fracs for r in ratios)
    _progress(f"[grid] {done}/{total} cells done; folds {run_folds}, organic fractions {fracs}, ratios {ratios}")

    for k in run_folds:
        test = org.loc[folds == k].reset_index(drop=True)
        train_all = org.loc[folds != k].reset_index(drop=True)
        u = hash_uniform(train_all[cols.id], f"organic|{sample_seed}")
        for g in fracs:
            keys = {r: ratio_key(k, g, r) for r in ratios}
            if all(store.done(x) for x in keys.values()):
                continue
            O_g = train_all.loc[u < g] if g < 1 else train_all
            t0 = time.time()
            teacher, rare = fit_model(O_g, hp, cols)
            t_test = score(teacher, test, cols, batch_size)
            base = {"fold": k, "organic_frac": g, "n_organic": int(len(O_g)), "n_test": int(len(test)),
                    "teacher_rows_after_rare_filter": rare["rows_after"],
                    **_teacher_summary(t_test, rule, len(cols.levels))}
            if not store.done(keys[0.0]):
                store.save(keys[0.0], t_test, {**base, "ratio_target": 0.0, "ratio": 0.0, "n_derived": 0,
                                               "capped": False, "seconds": round(time.time() - t0, 1)})
                done += 1
                _progress(f"  [{done}/{total}] fold {k} g={g}: teacher / ratio 0  "
                          f"L4={base[f'teacher L{len(cols.levels)} Accuracy']:.4f}  ({len(O_g):,} organic rows)")
            pending = [r for r in ratios if r > 0 and not store.done(keys[r])]
            if not pending:
                del teacher
                continue
            accepted, pinfo = pseudo_label(teacher, fd.pool, cols, rule, f"pool|{sample_seed}", batch_size)
            del teacher
            gc.collect()
            last_n, last_table = None, None
            for r in [r for r in ratios if r > 0]:
                n_target = int(round(r * len(O_g)))
                D = accepted.iloc[:n_target]
                if not store.done(keys[r]):
                    t1 = time.time()
                    if last_n == len(D) and last_table is not None:
                        table = last_table          # capped: same rows as the previous ratio
                    else:
                        student, _ = fit_model(pd.concat([O_g[cols.train_cols], D[cols.train_cols]],
                                                         ignore_index=True), hp, cols)
                        table = score(student, test, cols, batch_size)
                        del student
                    meta = {**base, **pinfo, "ratio_target": r, "ratio": len(D) / len(O_g),
                            "n_derived": int(len(D)), "capped": bool(len(D) < n_target),
                            "seconds": round(time.time() - t1, 1)}
                    store.save(keys[r], table, meta)
                    done += 1
                    _progress(f"  [{done}/{total}] fold {k} g={g} ratio {r} (realized {meta['ratio']:.3f}"
                              f"{', capped' if meta['capped'] else ''}): "
                              f"L4={table[f'L{len(cols.levels)} Correct'].mean():.4f}")
                    last_table = table
                else:
                    last_table = None
                last_n = len(D)
            del accepted
            gc.collect()
    return store.meta()


# ---------------------------------------------------------------------------
# Experiment 3a: teacher quality vs teacher size
# ---------------------------------------------------------------------------

def teacher_key(fold: int, frac: float) -> str:
    return f"teacher_f{fold}_g{_fmt(frac)}"


def pool_key(fold: int, frac: float) -> str:
    return f"pool_f{fold}_g{_fmt(frac)}"


def run_teacher_quality(store: Store, fd: FeedbackData, hp: HyperParams, cols: Cols, *, n_folds: int,
                        fold_seed: int, folds_to_run: Optional[Sequence[int]], teacher_fracs: Sequence[float],
                        rule: SelectionRule, sample_seed: int = 0, batch_size: int = 2000) -> pd.DataFrame:
    """Teacher trained on a share g of the organic rows outside fold k.

    Saves its predictions on fold k (confidence + correctness, for error and
    calibration) and its confidences on the pool (+ accepted flag).
    """
    org = fd.organic
    folds = fold_assignment(org, cols.id, n_folds, fold_seed)
    run_folds = list(range(n_folds)) if folds_to_run is None else list(folds_to_run)
    fracs = sorted({float(g) for g in teacher_fracs})
    L = len(cols.levels)
    for k in run_folds:
        test = org.loc[folds == k].reset_index(drop=True)
        train_all = org.loc[folds != k].reset_index(drop=True)
        u = hash_uniform(train_all[cols.id], f"organic|{sample_seed}")
        for g in fracs:
            if store.done(teacher_key(k, g)) and store.done(pool_key(k, g)):
                continue
            t0 = time.time()
            O_g = train_all.loc[u < g] if g < 1 else train_all
            teacher, _ = fit_model(O_g, hp, cols)
            t = score(teacher, test, cols, batch_size)
            pred = teacher.predict_frame(fd.pool[cols.cleaned], batch_size=batch_size)
            ptab = pd.DataFrame({f"L{d} Confidence": pred[f"L{d} Confidence"].to_numpy() for d in range(1, L + 1)})
            ptab["accepted"] = rule.mask(pred)
            ptab["derived"] = fd.pool["derived"].to_numpy()
            meta = {"fold": k, "teacher_frac": g, "n_organic": int(len(O_g)), "n_test": int(len(test)),
                    **_teacher_summary(t, rule, L), "pool_accept_share": float(ptab["accepted"].mean()),
                    "seconds": round(time.time() - t0, 1)}
            store.save(pool_key(k, g), ptab, {**meta, "table": "pool"})
            store.save(teacher_key(k, g), t, {**meta, "table": "teacher"})
            del teacher, pred
            gc.collect()
            _progress(f"  fold {k} g={g}: {len(O_g):,} rows, L{L} accuracy {meta[f'teacher L{L} Accuracy']:.4f}, "
                      f"L{L} error among accepted {meta[f'teacher L{L} error among accepted']:.4f}")
    return store.meta()


# ---------------------------------------------------------------------------
# Experiment 3b: noise dose (organic rows posing as the pool)
# ---------------------------------------------------------------------------

def dose_key(fold: int, dose: Optional[float]) -> str:
    return f"dose_f{fold}_" + ("without" if dose is None else f"rho{_fmt(dose)}")


def run_noise_dose(store: Store, fd: FeedbackData, hp: HyperParams, cols: Cols, *, n_folds: int,
                   fold_seed: int, folds_to_run: Optional[Sequence[int]], ratio: float, accept_rate: float,
                   doses: Sequence[float], rule: SelectionRule, sample_seed: int = 0,
                   oversize: float = 1.25, batch_size: int = 2000) -> pd.DataFrame:
    """For each fold k:

    1. split the organic rows outside fold k into a base B and a simulated pool
       S, sized so that after acceptance there are enough items for
       |S accepted| / |B| = `ratio` (|S| = oversize*ratio*|B|/accept_rate; the
       oversize margin covers an acceptance share on organic rows that is lower
       than on the real pool)
    2. teacher trained on B labels S; the rule is re-applied; capped at ratio*|B|
    3. S's true labels are known, so the teacher's errors are known. For dose
       rho, a nested random share rho of those errors keeps the teacher label
       and the rest get the true label: noise = rho * (teacher noise)
    4. students trained on B + S at each dose, and the teacher itself
       ('without'), are scored on fold k
    """
    if not (0 < accept_rate <= 1):
        raise ValueError("accept_rate must be in (0, 1]")
    if any(not (0 <= r <= 1) for r in doses):
        raise ValueError("doses must be in [0, 1] (shares of the teacher's own errors)")
    org = fd.organic
    folds = fold_assignment(org, cols.id, n_folds, fold_seed)
    run_folds = list(range(n_folds)) if folds_to_run is None else list(folds_to_run)
    doses = sorted({float(r) for r in doses})
    L = len(cols.levels)
    s_per_b = oversize * ratio / accept_rate
    share_s = s_per_b / (1 + s_per_b)
    for k in run_folds:
        keys = [dose_key(k, None)] + [dose_key(k, r) for r in doses]
        if all(store.done(x) for x in keys):
            continue
        test = org.loc[folds == k].reset_index(drop=True)
        train_all = org.loc[folds != k].reset_index(drop=True)
        u = hash_uniform(train_all[cols.id], f"dose_split|{sample_seed}")
        B = train_all.loc[u >= share_s].reset_index(drop=True)
        S = train_all.loc[u < share_s].reset_index(drop=True)
        t0 = time.time()
        teacher, _ = fit_model(B, hp, cols)
        t_test = score(teacher, test, cols, batch_size)
        pred = teacher.predict_frame(S[cols.cleaned], batch_size=batch_size)
        sel = rule.mask(pred)
        del teacher
        gc.collect()
        S = S.loc[sel].reset_index(drop=True)
        pred = pred.loc[sel].reset_index(drop=True)
        order = np.argsort(hash_uniform(S[cols.id], f"dose_pool|{sample_seed}"), kind="mergesort")
        n_cap = int(round(ratio * len(B)))
        order = order[:n_cap]
        S, pred = S.iloc[order].reset_index(drop=True), pred.iloc[order].reset_index(drop=True)
        # teacher errors per level (path prefix) and full-path errors
        pred_lab = pred[[f"Predicted {c}" for c in cols.levels]].astype(str).to_numpy()
        true_lab = S[cols.levels].astype(str).to_numpy()
        wrong_path = (pred_lab != true_lab).any(axis=1)
        eta_teacher = float(wrong_path.mean()) if len(S) else float("nan")
        err_idx = np.flatnonzero(wrong_path)
        err_order = err_idx[np.argsort(hash_uniform(S.loc[err_idx, cols.id], f"dose_err|{sample_seed}"),
                                       kind="mergesort")]
        base = {"fold": k, "n_base": int(len(B)), "n_sim_pool": int(len(S)),
                "ratio_target": ratio, "ratio": len(S) / max(len(B), 1),
                "sim_pool_accept_share": float(sel.mean()) if len(sel) else float("nan"),
                "teacher_noise": eta_teacher, "n_test": int(len(test)),
                **_teacher_summary(t_test, rule, L)}
        if not store.done(keys[0]):
            store.save(keys[0], t_test, {**base, "dose": None, "noise": None, "n_derived": 0,
                                         "seconds": round(time.time() - t0, 1)})
        _progress(f"  fold {k}: base {len(B):,}, simulated pool {len(S):,} (ratio {base['ratio']:.3f}), "
                  f"teacher path noise {eta_teacher:.4f}")
        for rho in doses:
            key = dose_key(k, rho)
            if store.done(key):
                continue
            t1 = time.time()
            keep_teacher = np.zeros(len(S), dtype=bool)
            keep_teacher[err_order[:int(round(rho * len(err_order)))]] = True
            labels = np.where(keep_teacher[:, None], pred_lab, true_lab)
            D = pd.DataFrame(labels, columns=cols.levels)
            D[cols.cleaned] = S[cols.cleaned].to_numpy()
            noise = float((labels != true_lab).any(axis=1).mean()) if len(S) else float("nan")
            student, _ = fit_model(pd.concat([B[cols.train_cols], D[cols.train_cols]], ignore_index=True), hp, cols)
            table = score(student, test, cols, batch_size)
            del student
            gc.collect()
            per_level = {f"noise L{d}": float((labels[:, :d] != true_lab[:, :d]).any(axis=1).mean())
                         for d in range(1, L + 1)} if len(S) else {}
            store.save(key, table, {**base, "dose": rho, "noise": noise, **per_level, "n_derived": int(len(S)),
                                    "seconds": round(time.time() - t1, 1)})
            _progress(f"  fold {k} dose {rho}: noise {noise:.4f}, L{L}={table[f'L{L} Correct'].mean():.4f}")
    return store.meta()


# ---------------------------------------------------------------------------
# Analysis: paired bootstrap over test rows
# ---------------------------------------------------------------------------

def bootstrap_cells(store: Store, cells: Sequence[Dict[int, str]], depth: int, n_boot: int = 300,
                    seed: int = 0, chunk: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Pooled accuracy at `depth` for each analysis cell, with a paired bootstrap.

    cells: one {fold: result key} per analysis cell. The cells of one fold
    share the same test rows, so rows are resampled once per fold and
    replicate, and differences between cells are paired.

    Returns (point estimates (n_cells,), bootstrap replicates (n_boot, n_cells)).
    """
    rng = np.random.default_rng(seed)
    folds = sorted({f for c in cells for f in c})
    hits = np.zeros((n_boot, len(cells)))
    point_hits = np.zeros(len(cells))
    total = 0
    for f in folds:
        if not all(f in c for c in cells):
            raise ValueError(f"fold {f} is not finished for every cell; set FOLDS_TO_RUN consistently")
        tables = [store.table(c[f]) for c in cells]
        rows0 = tables[0][ROW].to_numpy()
        for t in tables[1:]:
            if not np.array_equal(t[ROW].to_numpy(), rows0):
                raise ValueError("cells of one fold were scored on different test rows")
        M = np.column_stack([t[f"L{depth} Correct"].to_numpy(dtype=np.float32) for t in tables])
        n = len(M)
        total += n
        point_hits += M.sum(axis=0)
        for s in range(0, n_boot, chunk):
            e = min(s + chunk, n_boot)
            W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(e - s)]).astype(np.float32)
            hits[s:e] += W @ M
    return point_hits / total, hits / total


def summarize(point: np.ndarray, boot: np.ndarray, ci: float = 0.9) -> pd.DataFrame:
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2], axis=0)
    return pd.DataFrame({"estimate": point, "ci_low": lo, "ci_high": hi, "boot_se": boot.std(axis=0, ddof=1)})


# ---------------------------------------------------------------------------
# Analysis: curve fits
# ---------------------------------------------------------------------------

class RatioSurface:
    """Gain over ratio 0 as a smooth function of ratio r and organic size n:

        delta(r, n) = b1*z + b2*z^2 + b3*z*log(n/n_ref),   z = log(1 + r)

    No intercept: delta(0, n) = 0 by construction. Weighted least squares;
    `fit` accepts a matrix of y columns (bootstrap replicates) at once.
    """

    def __init__(self, n_ref: float):
        self.n_ref = float(n_ref)
        self.coef_: Optional[np.ndarray] = None

    def design(self, r, n) -> np.ndarray:
        z = np.log1p(np.asarray(r, dtype=float))
        ln = np.log(np.asarray(n, dtype=float) / self.n_ref)
        return np.column_stack([z, z ** 2, z * ln])

    def fit(self, r, n, y, w=None) -> "RatioSurface":
        X = self.design(r, n)
        w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
        sw = np.sqrt(w)[:, None]
        Y = np.asarray(y, dtype=float)
        Y = Y[:, None] if Y.ndim == 1 else Y
        self.coef_, *_ = np.linalg.lstsq(X * sw, Y * sw, rcond=None)
        return self

    def predict(self, r, n) -> np.ndarray:
        return self.design(np.atleast_1d(r), np.atleast_1d(n)) @ self.coef_


def fit_log_trend(n, y, w=None) -> np.ndarray:
    """y = a + b*log(n); returns [a, b] (or a 2 x k matrix for k y columns)."""
    X = np.column_stack([np.ones(len(n)), np.log(np.asarray(n, dtype=float))])
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
    sw = np.sqrt(w)[:, None]
    Y = np.asarray(y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    coef, *_ = np.linalg.lstsq(X * sw, Y * sw, rcond=None)
    return coef


def fit_power_law(n, y) -> Dict[str, Any]:
    """y = c + a * n^(-b) with c, a >= 0 (an error rate falling with training size).

    Falls back to a log-linear fit if the power law does not converge or has
    fewer than 4 points. Returns the parameters and a predict function.
    """
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(n) & np.isfinite(y)
    n, y = n[ok], y[ok]
    if len(np.unique(n)) >= 4:
        try:
            from scipy.optimize import curve_fit

            f = lambda x, c, a, b: c + a * np.power(x / n.max(), -b)
            p, _ = curve_fit(f, n, y, p0=[max(y.min() * 0.8, 1e-4), max(y.max() - y.min(), 1e-4), 0.5],
                             bounds=([0, 0, 0.01], [1, 10, 3]), maxfev=20000)
            return {"form": "power", "c": p[0], "a": p[1], "b": p[2],
                    "predict": lambda x, p=p, m=n.max(): p[0] + p[1] * np.power(np.asarray(x, float) / m, -p[2])}
        except Exception:
            pass
    coef = fit_log_trend(n, y).ravel()
    return {"form": "log-linear", "a": coef[0], "b": coef[1],
            "predict": lambda x, c=coef: c[0] + c[1] * np.log(np.asarray(x, float))}


def calibration_map(conf: np.ndarray, correct: np.ndarray, width: float = 0.05, min_items: int = 20):
    """(bucket centres, accuracy) for buckets with at least min_items items."""
    conf = np.clip(np.asarray(conf, float), 0, 1)
    nb = int(np.ceil(1 / width - 1e-9))
    idx = np.clip(np.floor(conf / width + 1e-12).astype(int), 0, nb - 1)
    n = np.bincount(idx, minlength=nb)
    h = np.bincount(idx, weights=np.asarray(correct, float), minlength=nb)
    keep = n >= min_items
    centres = (np.arange(nb) + 0.5) * width
    if not keep.any():
        return np.array([0.5]), np.array([h.sum() / max(n.sum(), 1)])
    return centres[keep], h[keep] / n[keep]


def apply_calibration(cmap, conf: np.ndarray) -> np.ndarray:
    """Expected accuracy of items at these confidences (linear between buckets)."""
    centres, acc = cmap
    return np.interp(np.clip(np.asarray(conf, float), 0, 1), centres, acc)
