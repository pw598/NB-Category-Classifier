"""Repository locations and timestamped folder helpers.

Every output lives in a timestamped subfolder so the top-level folders never
fill up with loose files:

    data/<ts>/                     raw_data_<ts>.txt, cleaned_data_<ts>.txt, metadata
    models/model_<ts>/             a trained model and everything needed to use it
    outputs/evaluation_<ts>/       cross-validation / grid-search checkpoints and results
    outputs/predictions_<ts>/      batch predictions and their metadata
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
QUERY_DIR = DATA_DIR / "queries"
CLEANING_DIR = REPO_ROOT / "cleaning"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def timestamp() -> str:
    """Current local time as YYYYMMDD_HHMMSS (sortable)."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def resolve(path: Optional[PathLike], base: Path = REPO_ROOT) -> Optional[Path]:
    """Resolve a user-supplied path. Relative paths are taken from the repo root."""
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


def _latest(parent: Path, prefix: str) -> Path:
    candidates = sorted(
        p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)
    ) if parent.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No '{prefix}*' folders in {parent}")
    return candidates[-1]


def find_folder(name: Optional[str], parent: Path, prefix: str) -> Path:
    """Find a timestamped folder.

    name may be None or 'latest' (most recent), a bare timestamp
    ('20261016_120000'), a folder name ('model_20261016_120000'), or a path.
    """
    if name is None or str(name).lower() == "latest":
        return _latest(parent, prefix)
    p = Path(name)
    if p.is_absolute() and p.is_dir():
        return p
    for candidate in (resolve(p), parent / str(name), parent / f"{prefix}{name}"):
        if candidate is not None and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find folder {name!r} under {parent}")


def new_folder(parent: Path, prefix: str, ts: Optional[str] = None) -> Path:
    ts = ts or timestamp()
    folder = parent / f"{prefix}{ts}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder
