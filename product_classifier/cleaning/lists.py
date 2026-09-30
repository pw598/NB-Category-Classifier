"""Loading the reference word lists used by the text cleaner.

Each file in cleaning_lists/ is a single-column .txt or .csv whose first
line is a header. The stem of the filename becomes the dictionary key, so
``colors.txt`` is reachable as ``lists["colors"]``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

DEFAULT_LIST_DIR = Path(__file__).parent / "cleaning_lists"

# Two columns rather than one, so it is not a reference list and is skipped
# by load_cleaning_lists. The name is fixed so nothing has to be configured.
EXPANSIONS_FILENAME = "abbrevs_to_expand.csv"


def load_cleaning_lists(directory: str | os.PathLike | None = None) -> Dict[str, List[str]]:
    """Read every .txt/.csv in `directory` into {stem: [values]}.

    Blank lines are skipped and the header row is dropped. Raises rather
    than silently returning an empty dict if the directory is missing,
    since a silently empty cleaning dict produces silently uncleaned text.
    """
    directory = Path(directory) if directory is not None else DEFAULT_LIST_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Cleaning-list directory not found: {directory}")

    data: Dict[str, List[str]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".txt", ".csv"):
            continue
        if path.name == EXPANSIONS_FILENAME:
            # Two columns, and a mapping rather than a list of terms to
            # remove. Reading it here would silently keep only column one.
            continue
        key = path.stem
        with path.open("r", encoding="utf-8-sig") as fh:
            if path.suffix.lower() == ".csv":
                reader = csv.reader(fh)
                next(reader, None)  # header
                values = [row[0].strip() for row in reader if row and row[0].strip()]
            else:
                values = [line.strip() for line in fh.readlines()[1:] if line.strip()]
        data[key] = values
    if not data:
        raise ValueError(f"No .txt or .csv files found in {directory}")
    return data


def load_expansions(
    directory: str | os.PathLike | None = None,
    required: bool = False,
) -> Dict[str, str]:
    """Read abbrevs_to_expand.csv into {abbreviation: expansion}.

    Two columns, header row dropped like every other list. Both sides are
    uppercased, because the descriptions are uppercase by the time this
    runs and matching is done against them literally.

    required=False returns {} when the file is absent, so the rest of the
    cleaning procedure works without it. Pass required=True when the
    caller has been told to expand -- a missing file should then be an
    error rather than a step that silently does nothing.
    """
    directory = Path(directory) if directory is not None else DEFAULT_LIST_DIR
    path = Path(directory) / EXPANSIONS_FILENAME

    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Abbreviation expansion was requested but {path} does not "
                "exist. Add it as a two-column CSV (abbreviation,expansion) "
                "with a header row, or turn expansion off."
            )
        return {}

    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for lineno, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                continue
            if len(row) < 2 or not row[1].strip():
                raise ValueError(
                    f"{path} line {lineno}: expected 'abbreviation,expansion' "
                    f"but found {row!r}. Every abbreviation needs something "
                    "to expand to."
                )
            out[row[0].strip().upper()] = row[1].strip().upper()

    if not out:
        if required:
            raise ValueError(f"{path} contains no abbreviation pairs.")
        return {}
    return out


def upper(values: List[str]) -> List[str]:
    return [v.upper() for v in values]


def sort_by_length(values: List[str]) -> List[str]:
    """Longest first.

    Matters for regex alternation: without this, 'BLU' would match inside
    'BLUE' and leave a stray 'E' behind.
    """
    return sorted(values, key=len, reverse=True)
