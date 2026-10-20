"""Saving and loading trained models (models/model_<ts>/).

Folder contents (same layout as NB-Category-Classifier-Minimized, plus
human-readable configuration files):

    vectorizer.pkl             fitted vectorizer (vocabulary, idf)
    model_<level>.pkl          one SparseMultinomialNB per level (sparse counts)
    path_index.pkl             level columns, class tables, valid-path index
    cleaner.pkl                the exact cleaning method (frozen code, term lists,
                               settings, English word list, learned vocabulary,
                               append fields)
    <cleaning method>/         readable copy of the cleaning-method folder
    vocabulary.txt             the step-11 vocabulary learned from the training data
    hierarchy_lookup.csv       level IDs -> names, from the training rows
    hyperparameters.json       every hyperparameter
    cleaning.json              readable summary of cleaner.pkl
    training_data.json         where the training data came from, row counts
    manifest.json              fingerprint, sizes, versions, file list
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from . import __version__, paths
from .cleaning import Cleaner
from .config import HyperParams
from .data import write_json
from .model import HierarchicalNB


def _tag(level: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", level).strip("_")


def _fingerprint(clf: HierarchicalNB) -> str:
    payload = json.dumps({
        "n_features": clf.n_features,
        "levels": clf.level_columns,
        "n_paths": clf.n_paths,
        "classes": [len(c) for c in clf.classes_],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _versions() -> Dict[str, str]:
    import numpy, scipy, sklearn

    return {"nbclassifier": __version__, "scikit-learn": sklearn.__version__,
            "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}


def save_model(
    folder: Path,
    clf: HierarchicalNB,
    cleaner: Cleaner,
    lookup: Optional[pd.DataFrame],
    training_metadata: Dict[str, Any],
    notes: str = "",
) -> Dict[str, Path]:
    import joblib

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Path] = {}

    files["vectorizer"] = folder / "vectorizer.pkl"
    joblib.dump(clf.vectorizer, files["vectorizer"], compress=3)
    for level, m in zip(clf.level_columns, clf.level_models):
        key = f"model_{_tag(level)}"
        files[key] = folder / f"{key}.pkl"
        joblib.dump(m, files[key], compress=3)
    files["path_index"] = folder / "path_index.pkl"
    joblib.dump({"level_columns": clf.level_columns, "classes_": clf.classes_,
                 "path_codes_": clf.path_codes_}, files["path_index"], compress=3)
    files["cleaner"] = folder / "cleaner.pkl"
    joblib.dump(cleaner, files["cleaner"], compress=3)
    files.update(cleaner.export(folder))  # copy of the cleaning-method folder + vocabulary.txt
    if lookup is not None:
        files["hierarchy_lookup"] = folder / "hierarchy_lookup.csv"
        lookup.drop_duplicates().to_csv(files["hierarchy_lookup"], index=False)
    files["hyperparameters"] = folder / "hyperparameters.json"
    write_json(files["hyperparameters"], clf.hp.to_dict())
    files["cleaning"] = folder / "cleaning.json"
    write_json(files["cleaning"], cleaner.describe())
    files["training_data"] = folder / "training_data.json"
    write_json(files["training_data"], training_metadata)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "folder": folder.name,
        "fingerprint": _fingerprint(clf),
        "level_columns": clf.level_columns,
        "n_features": clf.n_features,
        "n_paths": clf.n_paths,
        "classes_per_level": clf.classes_per_level,
        "n_training_rows": training_metadata.get("n_training_rows"),
        "prediction_mode": clf.hp.prediction_mode,
        "hyperparameters": clf.hp.to_dict(),
        "cleaning_method": cleaner.method_name,
        "versions": _versions(),
        "notes": notes,
    }
    files["manifest"] = folder / "manifest.json"
    manifest["files"] = {k: v.name for k, v in files.items()}
    write_json(files["manifest"], manifest)
    return files


@dataclass
class LoadedModel:
    folder: Path
    clf: HierarchicalNB
    cleaner: Cleaner
    lookup: Optional[pd.DataFrame]
    manifest: Dict[str, Any]
    training_metadata: Dict[str, Any]


def latest_model_dir() -> Path:
    return paths.find_folder("latest", paths.MODELS_DIR, prefix="model_")


def load_model(name: Optional[str] = None, verify: bool = True) -> LoadedModel:
    """Load models/model_<ts>/ (None or 'latest' = most recent)."""
    import joblib

    folder = paths.find_folder(name, paths.MODELS_DIR, prefix="model_")
    manifest = json.loads((folder / "manifest.json").read_text())
    hp = HyperParams.from_dict(json.loads((folder / "hyperparameters.json").read_text()))
    idx = joblib.load(folder / "path_index.pkl")
    clf = HierarchicalNB(idx["level_columns"], hp)
    clf.classes_, clf.path_codes_ = idx["classes_"], idx["path_codes_"]
    clf.vectorizer = joblib.load(folder / "vectorizer.pkl")
    clf.level_models = [joblib.load(folder / f"model_{_tag(l)}.pkl") for l in clf.level_columns]
    clf._derived = None
    clf.fitted_ = True
    if verify and _fingerprint(clf) != manifest["fingerprint"]:
        raise ValueError(f"Model in {folder} does not match its manifest fingerprint")
    cleaner = joblib.load(folder / "cleaner.pkl")
    lookup_path = folder / "hierarchy_lookup.csv"
    lookup = pd.read_csv(lookup_path, dtype=str) if lookup_path.is_file() else None
    tmeta_path = folder / "training_data.json"
    tmeta = json.loads(tmeta_path.read_text()) if tmeta_path.is_file() else {}
    return LoadedModel(folder, clf, cleaner, lookup, manifest, tmeta)
