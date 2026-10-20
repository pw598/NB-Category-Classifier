"""Description cleaning with a named cleaning method.

A cleaning method is a folder cleaning/<name>/ holding

    method.yaml        the settings (list files, step parameters)
    clean_steps.py     the step functions
    term_lists.py      list readers and pattern builders
    cleaning_lists/    units, colors, sizes, good tokens ...

build_cleaner(name) freezes that whole folder - code, lists and settings -
plus the English word list into one picklable Cleaner. The step code runs from
the frozen copy, so a dataset or model cleaned with a method keeps cleaning
the same way even if the folder is later edited.

Step 11 (rare-word removal) needs a vocabulary. Cleaning training data with
fit=True learns it (words in at least `min_word_descriptions` descriptions)
and stores it in the Cleaner; cleaning new data with fit=False re-uses it.
"""

from __future__ import annotations

import hashlib
import sys
import types
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from . import paths

METHOD_FILE = "method.yaml"
CODE_FILES = ("term_lists.py", "clean_steps.py")


def available_methods() -> List[str]:
    return sorted(p.parent.name for p in paths.CLEANING_DIR.glob(f"*/{METHOD_FILE}"))


def method_dir(name: str) -> Path:
    folder = paths.CLEANING_DIR / name
    if not (folder / METHOD_FILE).is_file():
        raise FileNotFoundError(
            f"Cleaning method {name!r} not found (expected {folder / METHOD_FILE}). "
            f"Available: {available_methods()}"
        )
    return folder


def _clean_marker_value(value: Any) -> Optional[str]:
    import re

    if value is None or (isinstance(value, float) and value != value):
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "<na>"):
        return None
    s = re.sub(r"[^0-9A-Za-z]+", "_", s.upper()).strip("_")
    return s or None


def _marker_spec(spec: Any, col: str = "?") -> Tuple[str, Optional[List[int]]]:
    """APPEND_FIELDS value -> (prefix, levels).

    "VEND"                                         -> VEND_<value>
    {"prefix": "UNSPSC", "levels": [2, 4, 6, 8]}   -> UNSPSC2_46 UNSPSC4_4618 UNSPSC6_461815 UNSPSC8_46181504
        (one token per level n: the first n characters of the value; levels the
        value is too short for are skipped)
    """
    import re

    if isinstance(spec, str):
        prefix, levels = spec, None
    elif isinstance(spec, dict) and "prefix" in spec:
        prefix, levels = spec["prefix"], spec.get("levels")
        unknown = set(spec) - {"prefix", "levels"}
        if unknown:
            raise ValueError(f"APPEND_FIELDS[{col!r}] has unknown key(s) {sorted(unknown)}")
        if levels is not None:
            if not (isinstance(levels, (list, tuple)) and levels
                    and all(isinstance(n, int) and n > 0 for n in levels)):
                raise ValueError(f"APPEND_FIELDS[{col!r}]['levels'] must be a list of positive integers")
            levels = sorted(set(int(n) for n in levels))
    else:
        raise ValueError(f"APPEND_FIELDS[{col!r}] must be a prefix string or "
                         "{'prefix': ..., 'levels': [...]}")
    if not re.fullmatch(r"[A-Za-z0-9]+", str(prefix)):
        raise ValueError(f"Marker prefix {prefix!r} for {col!r} must be letters/digits only")
    return str(prefix), levels


def _marker_tokens(value: Optional[str], prefix: str, levels: Optional[List[int]],
                   nulls_as_none: bool) -> str:
    if value is None:
        return f"{prefix}_None" if nulls_as_none else ""
    if not levels:
        return f"{prefix}_{value}"
    return " ".join(f"{prefix}{n}_{value[:n]}" for n in levels if len(value) >= n)


@dataclass
class Cleaner:
    """A frozen cleaning method (code + lists + settings + word lists)."""

    method_name: str
    method_yaml: str
    settings: Dict[str, Any]
    sources: Dict[str, str]                 # {"clean_steps.py": code, "term_lists.py": code}
    list_files: Dict[str, str]              # {file name: file text}, for provenance
    units: List[Dict[str, Any]]
    colors: List[str]
    sizes: List[str]
    good_tokens: List[str]
    english: Set[str] = field(default_factory=set)
    vocabulary: Optional[Dict[str, int]] = None        # step 11, learned with fit=True
    append_fields: Dict[str, Any] = field(default_factory=dict)  # {column: PREFIX or {"prefix", "levels"}}
    append_nulls_as_none: bool = True

    # -- frozen code ---------------------------------------------------------
    def _modules(self):
        mods = self.__dict__.get("_mods")
        if mods is None:
            tl = types.ModuleType(f"nbclassifier_frozen_{self.method_name}_term_lists")
            exec(compile(self.sources["term_lists.py"], f"<{self.method_name}/term_lists.py>", "exec"), tl.__dict__)
            cs = types.ModuleType(f"nbclassifier_frozen_{self.method_name}_clean_steps")
            # clean_steps.py does `import term_lists as tl`: point that import at the frozen copy
            previous = sys.modules.get("term_lists")
            sys.modules["term_lists"] = tl
            try:
                exec(compile(self.sources["clean_steps.py"], f"<{self.method_name}/clean_steps.py>", "exec"), cs.__dict__)
            finally:
                if previous is None:
                    sys.modules.pop("term_lists", None)
                else:
                    sys.modules["term_lists"] = previous
            patterns = tl.unit_patterns(self.units)
            mods = (cs, tl, patterns)
            self.__dict__["_mods"] = mods
        return mods

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_mods", None)
        return state

    # -- cleaning ------------------------------------------------------------
    def clean_text(self, series: pd.Series, fit: bool = False, verbose: bool = False) -> Tuple[pd.Series, Dict[str, Any]]:
        """Run steps 1-13. Row-preserving.

        fit=True learns the step-11 vocabulary from these descriptions (training
        data); fit=False uses the stored vocabulary (new data).
        Returns (cleaned Series, report).
        """
        cs, tl, patterns = self._modules()
        s = self.settings
        report: Dict[str, Any] = {}

        def step(label, before, after):
            if verbose:
                cs.show_changes(label, before, after)
            return after

        # Steps 1-2: uppercase, into a new field
        text = series.fillna("").astype(str).str.upper().tolist()
        # Step 3: repair split words
        text = step("Step 3, split words", text, cs.repair_split_words(text, s["field_width"], s["repair_min_count"]))
        # Step 4: remove markers
        text = step("Step 4, markers", text, cs.remove_markers(text, s["markers"]))
        # Step 5: first word
        new, removed, kept = cs.strip_first_word(
            text, self.english, self.good_tokens, patterns,
            s["keep_word_forms"], s["keep_measurements"], s["keep_common_words"])
        text = step("Step 5, first word", text, new)
        report["first_word_kept_by_safeguard"] = dict(kept)
        report["first_word_most_removed"] = dict(removed.most_common(25))
        if verbose:
            print("    Kept by the safeguards:", dict(kept))
            print("    Most often removed:", ", ".join(f"{w} ({n:,})" for w, n in removed.most_common(25)))
        # Steps 6-10
        text = step("Step 6, colors", text, cs.replace_colors(text, self.colors))
        text = step("Step 7, units", text, cs.standardize_units(text, patterns))
        text = step("Step 8, gluing", text, cs.glue_units(text, patterns))
        text = step("Step 9, sizes", text, cs.replace_sizes(text, self.sizes, patterns))
        text = step("Step 10, punctuation", text, cs.remove_punctuation(text, s["punctuation_mode"]))
        # Step 11: rare words
        if fit:
            self.vocabulary = cs.build_vocabulary(text, s["min_word_descriptions"])
            if verbose:
                print(f"Vocabulary: {len(self.vocabulary):,} words in at least "
                      f"{s['min_word_descriptions']} descriptions (learned from this data).")
        elif self.vocabulary is None:
            raise RuntimeError("This cleaner has no vocabulary yet: clean the training data with fit=True first")
        elif verbose:
            print(f"Vocabulary: {len(self.vocabulary):,} words (from the training data).")
        report["vocabulary_size"] = len(self.vocabulary)
        text = step("Step 11, rare words", text, cs.remove_rare_words(text, self.vocabulary))
        # Step 12: repeated placeholders
        text = step("Step 12, repeated placeholders", text,
                    cs.collapse_repeats(text, s.get("collapse_placeholders", ["<SIZE>", "<COLOR>"])))
        # Step 13: spaces
        text = cs.tidy_spaces(text)
        return pd.Series(text, index=series.index, dtype=object), report

    def clean_frame(self, df: pd.DataFrame, text_col: str, fit: bool = False,
                    verbose: bool = False) -> Tuple[pd.Series, Dict[str, Any]]:
        """clean_text() on df[text_col], then append marker tokens (e.g. VEND_<value>)
        from APPEND_FIELDS. Rows that clean to nothing get no markers."""
        cleaned, report = self.clean_text(df[text_col], fit=fit, verbose=verbose)
        if not self.append_fields:
            return cleaned, report
        missing = [c for c in self.append_fields if c not in df.columns]
        if missing:
            raise KeyError(f"Append-field column(s) {missing} not in the data. "
                           f"This cleaner appends {self.append_fields}.")
        markers = []
        for col, spec in self.append_fields.items():
            prefix, levels = _marker_spec(spec)
            vals = df[col].map(_clean_marker_value)
            markers.append(vals.map(lambda v, p=prefix, lv=levels: _marker_tokens(
                v, p, lv, self.append_nulls_as_none)).astype(str))
        suffix = markers[0]
        for m in markers[1:]:
            suffix = suffix + " " + m
        suffix = suffix.str.replace(r"\s+", " ", regex=True).str.strip()
        has_text = cleaned.str.len() > 0
        return cleaned.where(~has_text, (cleaned + " " + suffix).str.strip()), report

    # -- provenance ----------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """JSON-serialisable summary (written next to datasets and models)."""
        code = "".join(self.sources[f] for f in CODE_FILES)
        return {
            "cleaning_method": self.method_name,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest()[:16],
            "settings": self.settings,
            "list_files": sorted(self.list_files),
            "n_units": len(self.units), "n_colors": len(self.colors), "n_sizes": len(self.sizes),
            "n_good_tokens": len(self.good_tokens),
            "english_words": len(self.english),
            "vocabulary_size": None if self.vocabulary is None else len(self.vocabulary),
            "append_fields": self.append_fields,
            "append_nulls_as_none": self.append_nulls_as_none,
        }

    def export(self, folder: Path) -> Dict[str, Path]:
        """Write a readable copy of the frozen method and the learned vocabulary."""
        folder = Path(folder)
        out = {}
        mdir = folder / self.method_name
        (mdir / "cleaning_lists").mkdir(parents=True, exist_ok=True)
        (mdir / METHOD_FILE).write_text(self.method_yaml, encoding="utf-8")
        for name, code in self.sources.items():
            (mdir / name).write_text(code, encoding="utf-8")
        for name, txt in self.list_files.items():
            (mdir / "cleaning_lists" / name).write_text(txt, encoding="utf-8")
        out["cleaning_method"] = mdir
        if self.vocabulary is not None:
            cs, _, _ = self._modules()
            out["vocabulary"] = folder / "vocabulary.txt"
            cs.save_vocabulary(self.vocabulary, out["vocabulary"])
        return out

    def __repr__(self) -> str:
        return (f"Cleaner(method={self.method_name!r}, english_words={len(self.english):,}, "
                f"vocabulary={'not learned' if self.vocabulary is None else f'{len(self.vocabulary):,} words'}, "
                f"append_fields={self.append_fields})")


def build_cleaner(
    method: str,
    append_fields: Optional[Dict[str, Any]] = None,
    append_nulls_as_none: bool = True,
    english: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Cleaner:
    """Freeze cleaning/<method>/ into a Cleaner (vocabulary not yet learned).

    english: override the English word list (tests); normally None, which uses
    the method's english_words_file or the NLTK list (installed if missing).
    """
    import re

    import yaml

    folder = method_dir(method)
    method_yaml = (folder / METHOD_FILE).read_text(encoding="utf-8")
    settings = yaml.safe_load(method_yaml)
    sources = {f: (folder / f).read_text(encoding="utf-8") for f in CODE_FILES}
    lists_dir = folder / "cleaning_lists"
    list_files = {p.name: p.read_text(encoding="utf-8-sig") for p in sorted(lists_dir.iterdir()) if p.is_file()}

    cleaner = Cleaner(method_name=settings.get("name", method), method_yaml=method_yaml, settings=settings,
                      sources=sources, list_files=list_files, units=[], colors=[], sizes=[], good_tokens=[])
    # parse the lists with the method's own term_lists.py
    tl = types.ModuleType("tl_tmp")
    exec(compile(sources["term_lists.py"], f"<{method}/term_lists.py>", "exec"), tl.__dict__)
    L = settings["lists"]
    cleaner.units = tl.read_units(lists_dir / L["units"])
    cleaner.colors = tl.read_terms(lists_dir / L["colors"]) + tl.read_terms(lists_dir / L["color_abbrevs"])
    cleaner.sizes = tl.read_terms(lists_dir / L["sizes"])
    cleaner.good_tokens = tl.read_terms(lists_dir / L["good_tokens"])

    if english is None:
        words_file = settings.get("english_words_file")
        if words_file:
            english = tl.english_words(folder / words_file)
        else:
            from .bootstrap import ensure_word_corpus

            english = ensure_word_corpus(verbose=verbose)
    cleaner.english = set(english)

    append_fields = dict(append_fields or {})
    for col, spec in append_fields.items():
        _marker_spec(spec, col)  # validates
    cleaner.append_fields = append_fields
    cleaner.append_nulls_as_none = append_nulls_as_none
    cleaner._modules()  # compile now so errors surface here
    if verbose:
        print(f"{cleaner}\n  lists: {len(cleaner.units)} units, {len(cleaner.colors)} colors and abbreviations, "
              f"{len(cleaner.sizes)} sizes, {len(cleaner.good_tokens):,} good tokens")
    return cleaner


# ---------------------------------------------------------------------------
# Vocabulary statistics
# ---------------------------------------------------------------------------

def ngram_stats(texts: pd.Series) -> Dict[str, int]:
    """Unique words, adjacent word pairs, and both, in the cleaned text
    (words = whitespace-separated, as the cleaning method defines them)."""
    uni: Set[str] = set()
    bi: Set[Tuple[str, str]] = set()
    for doc in texts.fillna("").astype(str):
        toks = doc.split()
        uni.update(toks)
        bi.update(zip(toks, toks[1:]))
    return {
        "unique_unigrams": len(uni),
        "unique_bigrams": len(bi),
        "unique_unigrams_plus_bigrams": len(uni) + len(bi),
    }


def top_tokens(texts: pd.Series, n: int = 25) -> pd.DataFrame:
    c: Counter = Counter()
    for doc in texts.fillna("").astype(str):
        c.update(doc.split())
    return pd.DataFrame(c.most_common(n), columns=["token", "count"])
