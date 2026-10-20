"""Install-on-demand helpers (NLTK).

Called by the cleaning code when a cleaning method needs the English word list, so
notebooks never need a pip cell. Both functions are no-ops once the package
and corpus are present.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Set

NLTK_REQUIREMENT = "nltk>=3.8"


def ensure_nltk(verbose: bool = True):
    """Import nltk, pip-installing it first if it is missing."""
    try:
        import nltk  # noqa: F401
    except ImportError:
        if verbose:
            print(f"[bootstrap] nltk not found; installing {NLTK_REQUIREMENT} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", NLTK_REQUIREMENT]
        )
        importlib.invalidate_caches()
    import nltk

    return nltk


def ensure_word_corpus(verbose: bool = True) -> Set[str]:
    """The NLTK 'words' corpus as a lowercase set, downloading it if needed."""
    nltk = ensure_nltk(verbose=verbose)
    try:
        from nltk.corpus import words

        vocab = words.words()
    except LookupError:
        if verbose:
            print("[bootstrap] downloading NLTK 'words' corpus ...")
        nltk.download("words", quiet=True)
        from nltk.corpus import words

        vocab = words.words()
    return {w.lower() for w in vocab}
