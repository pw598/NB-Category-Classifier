"""Cross-validation and grid search with on-disk checkpoints.

An evaluation lives in outputs/evaluation_<ts>/:

    evaluation_config.json         CV settings, columns, candidate list
    dataset/data.parquet           the exact cleaned rows evaluated (so a resume
                                   never depends on re-pulling SQL)
    dataset/cleaner.pkl, metadata.json
    predictions/candidate_<k>/split_<s>.parquet
                                   out-of-sample predictions for one candidate
                                   on one fold/repeat, written as soon as done
    results.csv, split_metrics.csv summary tables (rebuilt from the parquet files)
    best/                          exported predictions, bucket tables and plots
                                   for the best candidate

Memory: only one fitted model is alive at a time, and per row only the
predicted path, per-level confidence/margin and input-quality columns are
kept - never the (rows x paths) score matrix. Candidates that share every
vectorizer setting and the rare-path threshold share one vectorizer and one
set of per-level counts per fold; alpha, fit_prior and prediction_mode are
applied to those counts without refitting.

Splits are made once on the full dataset. The rare-path threshold is applied
to each split's training rows only; every held-out row is predicted.

Resume: point RESUME_FROM at the folder. Finished (candidate, split) files
are skipped. New candidates in the notebook's grid are appended; CV settings
must match the original run.
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import paths
from .config import CVSettings, HyperParams
from .data import PreparedData, filter_rare_paths, write_json
from .evaluation import headline
from .folds import holdout_splits, id_hash_folds, id_hash_sample, stratified_folds
from .model import HierarchicalNB
from .predict import accuracy_by_level, attach_names

ROW_COL = "_row"


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


@dataclass
class Columns:
    text_col: str
    cleaned_col: str
    level_columns: List[str]
    name_columns: List[str]
    id_cols: List[str]

    def to_dict(self):
        return dict(self.__dict__)


class Evaluation:
    """One evaluation folder. Create with Evaluation.create(...) or Evaluation.open(...)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        cfg = json.loads((self.folder / "evaluation_config.json").read_text())
        self.config = cfg
        self.cv = CVSettings(**cfg["cv"])
        self.cols = Columns(**cfg["columns"])
        self.candidates = [HyperParams.from_dict(c) for c in cfg["candidates"]]
        self.grid_search = cfg["grid_search"]
        self._data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    @property
    def split_col(self) -> str:
        return "fold" if self.cv.cv_type == "oof" else "repeat"

    @property
    def n_splits(self) -> int:
        return self.cv.n_folds if self.cv.cv_type == "oof" else self.cv.n_repeats

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = pd.read_parquet(self.folder / "dataset" / "data.parquet")
        return self._data

    def _pred_path(self, cand: int, split: int) -> Path:
        return self.folder / "predictions" / f"candidate_{cand:03d}" / f"split_{split}.parquet"

    def is_done(self, cand: int, split: int) -> bool:
        return self._pred_path(cand, split).is_file()

    # ------------------------------------------------------------------
    @classmethod
    def create(cls, prepared: PreparedData, candidates: Sequence[HyperParams], cv: CVSettings,
               columns: Columns, grid_search: bool, folder: Optional[Path] = None,
               sample_frac: Optional[float] = None, sample_seed: int = 0) -> "Evaluation":
        """sample_frac: evaluate on about this share of rows only (for exploring a grid
        quickly). Rows are chosen by a hash of FOLD_KEY_COL, so the NB and NN repos
        sample the same items for the same SAMPLE_FRAC / SAMPLE_SEED."""
        import joblib

        folder = folder or paths.new_folder(paths.OUTPUTS_DIR, "evaluation_")
        (folder / "dataset").mkdir(parents=True, exist_ok=True)
        df = prepared.df
        keep = list(OrderedDict.fromkeys(
            [c for c in columns.id_cols if c in df.columns]
            + ([cv.fold_key_col] if cv.fold_key_col in df.columns else [])
            + columns.level_columns
            + [c for c in columns.name_columns if c in df.columns]
            + [columns.text_col, columns.cleaned_col]
        ))
        if cv.cv_type == "oof" and cv.fold_method == "id_hash" and cv.fold_key_col not in df.columns:
            raise KeyError(f"FOLD_KEY_COL {cv.fold_key_col!r} not in data columns {list(df.columns)}")
        data = df[keep].copy()
        n_before = len(data)
        if sample_frac is not None and sample_frac < 1:
            key = cv.fold_key_col if cv.fold_key_col in data.columns else columns.id_cols[0]
            data = data.loc[id_hash_sample(data[key], sample_frac, sample_seed)].reset_index(drop=True)
            print(f"[evaluation] SAMPLE_FRAC={sample_frac}: {len(data):,} of {n_before:,} rows kept")
        data[ROW_COL] = np.arange(len(data))
        _atomic_parquet(data, folder / "dataset" / "data.parquet")
        joblib.dump(prepared.cleaner, folder / "dataset" / "cleaner.pkl", compress=3)
        write_json(folder / "dataset" / "metadata.json", prepared.metadata)
        cfg = {
            "created": folder.name.replace("evaluation_", ""),
            "grid_search": bool(grid_search),
            "cv": cv.to_dict(),
            "columns": columns.to_dict(),
            "candidates": [c.to_dict() for c in candidates],
            "n_rows": int(len(data)),
            "sample_frac": sample_frac,
            "sample_seed": sample_seed if sample_frac is not None else None,
            "data_source": prepared.metadata.get("source"),
            "cleaning": prepared.metadata.get("cleaning"),
        }
        write_json(folder / "evaluation_config.json", cfg)
        print(f"[evaluation] created {folder}")
        return cls(folder)

    @classmethod
    def open(cls, name: str, candidates: Optional[Sequence[HyperParams]] = None,
             cv: Optional[CVSettings] = None) -> "Evaluation":
        folder = paths.find_folder(name, paths.OUTPUTS_DIR, prefix="evaluation_")
        ev = cls(folder)
        if cv is not None and cv.relevant() != ev.cv.relevant():
            raise ValueError(
                "CV settings differ from the run being resumed.\n"
                f"  saved   : {ev.cv.relevant()}\n  notebook: {cv.relevant()}"
            )
        if candidates:
            have = {json.dumps(c.effective(), sort_keys=True) for c in ev.candidates}
            new = [c for c in candidates if json.dumps(c.effective(), sort_keys=True) not in have]
            if new:
                ev.config["candidates"] += [c.to_dict() for c in new]
                ev.candidates += list(new)
                write_json(folder / "evaluation_config.json", ev.config)
                print(f"[evaluation] appended {len(new)} new candidate(s) from the notebook grid")
        done = sum(ev.is_done(c, s) for c in range(len(ev.candidates)) for s in range(ev.n_splits))
        total = len(ev.candidates) * ev.n_splits
        print(f"[evaluation] resuming {folder.name}: {done}/{total} (candidate, split) fits done")
        return ev

    # ------------------------------------------------------------------
    def _splits(self, df: pd.DataFrame) -> List[Tuple[int, np.ndarray]]:
        cv = self.cv
        if cv.cv_type == "oof":
            if cv.fold_method == "id_hash":
                folds = id_hash_folds(df[cv.fold_key_col], cv.n_folds, cv.fold_seed)
            else:
                folds = stratified_folds(df[self.cols.level_columns[-1]], cv.n_folds, cv.fold_seed)
            return [(f, folds == f) for f in range(cv.n_folds)]
        return list(holdout_splits(len(df), cv.n_repeats, cv.test_size, cv.holdout_seed))

    def run(self, tracker=None, batch_size: int = 2000) -> pd.DataFrame:
        """Fit/predict every outstanding (candidate, split); returns results()."""
        cols, levels = self.cols, self.cols.level_columns
        groups: "OrderedDict[str, List[int]]" = OrderedDict()
        for i, hp in enumerate(self.candidates):
            groups.setdefault(hp.vectorizer_key(), []).append(i)

        total = len(self.candidates) * self.n_splits
        done = sum(self.is_done(c, s) for c in range(len(self.candidates)) for s in range(self.n_splits))
        # Splits are made on the full dataset, identical for every candidate. The rare-path
        # threshold is applied to the TRAINING part only (as in deployment); every held-out
        # row is predicted, so the OOF file covers all rows and candidates stay comparable.
        splits = self._splits(self.data)
        for key, cids in groups.items():
            if all(self.is_done(c, s) for c in cids for s in range(self.n_splits)):
                continue
            base = self.candidates[cids[0]]
            print(f"\n[group] candidates {cids} | rare_path_threshold={base.rare_path_threshold}")
            for split, test in splits:
                pending = [c for c in cids if not self.is_done(c, split)]
                if not pending:
                    continue
                t0 = time.time()
                held = self.data.loc[test].reset_index(drop=True)
                train, info = filter_rare_paths(self.data.loc[~test], levels, base.rare_path_threshold)
                print(f"  {self.split_col} {split}: training on {info['rows_after']:,} rows / "
                      f"{info['paths_after']:,} paths ({info['rows_dropped']:,} rare-path rows dropped); "
                      f"predicting {len(held):,} held-out rows")
                if info["rows_after"] == 0:
                    raise ValueError(f"rare_path_threshold={base.rare_path_threshold} leaves no training rows")
                clf =HierarchicalNB(levels, base).fit(train[cols.cleaned_col], train[levels])
                X = clf.transform(held[cols.cleaned_col])
                quality = clf.input_quality(held[cols.cleaned_col])
                fit_s = time.time() - t0
                del train
                ident = held[[c for c in cols.id_cols if c in held.columns] + [ROW_COL]]
                actual = held[levels].rename(columns={c: f"Actual {c}" for c in levels})
                for c in pending:
                    hp = self.candidates[c]
                    t1 = time.time()
                    m = clf.clone_with(hp.alpha, hp.fit_prior, hp.prediction_mode)
                    pred = m.predict_frame(X=X, batch_size=batch_size)
                    table = pd.concat([ident, pd.DataFrame({self.split_col: np.full(len(held), split)}),
                                       actual, pred, quality], axis=1)
                    path = self._pred_path(c, split)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_parquet(table, path)
                    acc = accuracy_by_level(table, levels)
                    done += 1
                    print(f"  [{done}/{total}] candidate {c} {self.split_col} {split}: "
                          + " ".join(f"{k.split()[0]}={v:.4f}" for k, v in acc.items())
                          + f"  (fit {fit_s:.0f}s, predict {time.time() - t1:.0f}s, "
                            f"{clf.n_features:,} features, {clf.n_paths:,} paths)")
                    del m, pred, table
                del clf, X, quality, held
                gc.collect()

        results = self.results(refresh=True)
        if tracker is not None:
            self._log(tracker, results)
        return results

    # ------------------------------------------------------------------
    def split_metrics(self) -> pd.DataFrame:
        levels = self.cols.level_columns
        need = [f"Predicted {c}" for c in levels] + [f"Actual {c}" for c in levels]
        rows = []
        for c in range(len(self.candidates)):
            for s in range(self.n_splits):
                p = self._pred_path(c, s)
                if p.is_file():
                    t = pd.read_parquet(p, columns=need)
                    rows.append({"candidate": c, self.split_col: s, "n_rows": len(t),
                                 **accuracy_by_level(t, levels)})
        return pd.DataFrame(rows)

    def results(self, refresh: bool = False) -> pd.DataFrame:
        path = self.folder / "results.csv"
        if path.is_file() and not refresh:
            return pd.read_csv(path).set_index("candidate")
        sm = self.split_metrics()
        sm.to_csv(self.folder / "split_metrics.csv", index=False)
        params = pd.DataFrame([{**hp.to_dict(), "ngram_range": str(tuple(hp.ngram_range))}
                               for hp in self.candidates])
        params.index.name = "candidate"
        if sm.empty:
            return params
        acc_cols = [c for c in sm.columns if c.endswith("Accuracy")]
        agg = sm.groupby("candidate").agg(
            splits_done=(self.split_col, "nunique"), n_rows=("n_rows", "sum"),
            **{f"{c}": (c, "mean") for c in acc_cols},
            **{f"{c} std": (c, "std") for c in acc_cols},
        )
        out = params.join(agg, how="left")
        out["complete"] = out["splits_done"].fillna(0).astype(int) == self.n_splits
        out.to_csv(path)
        return out

    def best(self, metric: str = "L4 Accuracy") -> Tuple[int, HyperParams, pd.Series]:
        res = self.results(refresh=True)
        if "complete" not in res.columns or not res["complete"].fillna(False).astype(bool).any():
            raise RuntimeError("No candidate has finished every split yet")
        res = res[res["complete"].fillna(False).astype(bool)]
        if res.empty:
            raise RuntimeError("No candidate has finished every split yet")
        if metric not in res.columns:
            raise KeyError(f"SELECTION_METRIC {metric!r} not in {list(res.columns)}")
        cid = int(res[metric].idxmax())
        return cid, self.candidates[cid], res.loc[cid]

    def predictions(self, cand: int, with_text: bool = True) -> pd.DataFrame:
        parts = [pd.read_parquet(self._pred_path(cand, s))
                 for s in range(self.n_splits) if self.is_done(cand, s)]
        pred = pd.concat(parts, ignore_index=True)
        if with_text:
            text = self.data[[ROW_COL, self.cols.text_col, self.cols.cleaned_col]]
            pred = pred.merge(text, on=ROW_COL, how="left")
        return pred.sort_values([self.split_col, ROW_COL]).reset_index(drop=True)

    def headline(self, cand: int) -> pd.DataFrame:
        pred = self.predictions(cand, with_text=False)
        return headline(pred, self.cols.level_columns, split_col=self.split_col)

    def export(self, cand: int, folder: Optional[Path] = None) -> Path:
        """Write the candidate's predictions (ensemble format) plus a JSON sidecar."""
        folder = Path(folder or self.folder / "best")
        folder.mkdir(parents=True, exist_ok=True)
        pred = self.predictions(cand).drop(columns=[ROW_COL])
        # names, as in batch predictions (lookup built from the evaluated rows)
        lvl, names = self.cols.level_columns, [c for c in self.cols.name_columns if c in self.data.columns]
        if names:
            lookup = self.data[lvl + names].drop_duplicates()
            pred = attach_names(pred, lookup, lvl, names)
        out = folder / ("oof_predictions.csv" if self.cv.cv_type == "oof" else "holdout_predictions.csv")
        pred.to_csv(out, index=False)
        hp = self.candidates[cand]
        sidecar = {
            "model": "naive_bayes",
            "evaluation": self.folder.name,
            "candidate": cand,
            "n_rows": int(len(pred)),
            "cv_type": self.cv.cv_type,
            "sample_frac": self.config.get("sample_frac"),
            "sample_seed": self.config.get("sample_seed"),
            "n_folds": self.cv.n_folds if self.cv.cv_type == "oof" else None,
            "fold_seed": self.cv.fold_seed,
            "fold_key_col": self.cv.fold_key_col,
            "fold_method": self.cv.fold_method,
            "level_columns": self.cols.level_columns,
            "prediction_mode": hp.prediction_mode,
            "hyperparameters": hp.to_dict(),
            "cleaning": self.config.get("cleaning"),
            "data_source": self.config.get("data_source"),
        }
        write_json(out.with_suffix(".json"), sidecar)
        print(f"[export] {len(pred):,} rows -> {out}")
        return out

    # ------------------------------------------------------------------
    def _log(self, tracker, results: pd.DataFrame) -> None:
        tracker.log_params({"cv": self.cv.to_dict(), "grid_search": self.grid_search,
                            "n_candidates": len(self.candidates), "evaluation_folder": self.folder.name})
        tracker.log_table(results.reset_index(), "results.csv")
        sm = self.split_metrics()
        for c, hp in enumerate(self.candidates):
            row = results.loc[c] if c in results.index else None
            with tracker.child(f"candidate_{c:03d}"):
                tracker.log_params(hp.to_dict())
                if row is not None:
                    tracker.log_metrics({k: row[k] for k in results.columns if "Accuracy" in k})
                for _, r in sm[sm["candidate"] == c].iterrows():
                    tracker.log_metrics({k: r[k] for k in sm.columns if k.endswith("Accuracy")},
                                        step=int(r[self.split_col]), prefix="split ")
