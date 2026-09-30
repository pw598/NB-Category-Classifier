"""Environment-aware default paths.

Data and outputs live next to the repo, on a cluster and on a laptop
alike, so nothing has to be edited by hand when the notebook moves.
Absolute cluster paths are deliberately not used: they look reasonable
until a workspace mount refuses the write, and that failure surfaces at
the end of a long run rather than the start.

Override either root with an environment variable if the defaults are
wrong for your setup:

    PRODUCT_CLASSIFIER_DATA_DIR
    PRODUCT_CLASSIFIER_OUTPUT_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

# <repo>/product_classifier/paths.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent



def on_databricks() -> bool:
    """True when running on a Databricks cluster with /dbfs mounted."""
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return True
    return os.path.isdir("/dbfs")


def data_dir() -> Path:
    """Directory holding the golden set."""
    override = os.environ.get("PRODUCT_CLASSIFIER_DATA_DIR")
    if override:
        return Path(override)
    # Always repo-relative, on Databricks and locally alike: the data ships
    # with the repo, so <repo>/data is correct in both places.
    return REPO_ROOT / "data"


def output_dir() -> Path:
    """Directory to write run outputs into.

    Repo-relative everywhere, on a cluster and locally alike. Writing to
    /dbfs/FileStore looks reasonable until the workspace mount refuses it,
    and the failure only shows up at the end of a long run.
    """
    override = os.environ.get("PRODUCT_CLASSIFIER_OUTPUT_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "outputs"


def data_path(filename: str = "cleaned_data.txt") -> str:
    return str(data_dir() / filename)


def describe() -> str:
    env = "Databricks" if on_databricks() else "local"
    return (
        f"environment: {env}\n"
        f"  data dir:   {data_dir()}\n"
        f"  output dir: {output_dir()}"
    )
