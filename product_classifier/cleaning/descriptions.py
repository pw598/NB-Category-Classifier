"""The description-cleaning procedure used to build the golden set.

A faithful port of the original notebook procedure, which is considerably
more aggressive than cleaning.text.CleaningConfig: as well as stripping
colours, sizes and units, it removes pack counts, dimension strings,
fractions, standalone numbers and all remaining punctuation. What survives
is close to bare nouns and adjectives, which is the point -- a category is
determined by what a thing *is*, not by how big it is or how many come in
the box.

Two steps in here have no equivalent in CleaningConfig:

first-word stripping
    Descriptions frequently open with a vendor code or part number
    ('QJX8A TORQUE HEAD BOXEND'). If the first token is not a recognisable
    English word it is dropped, on the reasoning that it identifies the
    individual product rather than its category -- and a token unique to
    one SKU is noise the model would otherwise try to learn from.

the two-phase regex pass
    Order matters and the phases are not interchangeable. Phase 1 removes
    the compound patterns while the punctuation that delimits them is
    still present -- '1-1/4X3-1/2' is only recognisable as a dimension
    while it still has its slashes. Phase 2 then removes the reference-list
    terms and finally the punctuation itself. Run the other way round, the
    dimension strings fragment into stray digits that the earlier patterns
    would no longer match.

Note the text is *not* uppercased here, matching the original. The
patterns and the reference lists are uppercase, so this assumes the source
descriptions already are -- true of the ERP extract. Pass
uppercase=True if that stops holding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .lists import load_cleaning_lists, load_expansions, upper

DEFAULT_ORIG_COL = "FullDesc_orig"
FIRST_WORD_COL = "FirstWordReal"

# Units that may legitimately trail a number inside a dimension string, as
# opposed to the tokens that make up the dimension itself.
SPEC_UNITS = ["FT", "IN", "CM", "MM", "MIL", "TPI", "TP", "T"]


# ---------------------------------------------------------------------------
# Primitives, on a whole frame
# ---------------------------------------------------------------------------

def regex_deletion(df: pd.DataFrame, column: str, patterns: Sequence[str]) -> pd.DataFrame:
    """Delete every match of every pattern, in order, in place on a copy."""
    out = df.copy()
    series = out[column].astype(str)
    for pattern in patterns:
        series = series.str.replace(pattern, "", regex=True)
    out[column] = series
    return out


def regex_replacement(
    df: pd.DataFrame, column: str, pattern_replacements: Dict[str, str]
) -> pd.DataFrame:
    """Apply {pattern: replacement} substitutions in dict order."""
    out = df.copy()
    series = out[column].astype(str)
    for pattern, replacement in pattern_replacements.items():
        series = series.str.replace(pattern, replacement, regex=True)
    out[column] = series
    return out


# ---------------------------------------------------------------------------
# First-word stripping
# ---------------------------------------------------------------------------

def load_english_vocabulary(download: bool = True) -> set:
    """The NLTK English word list, lowercased.

    Downloading needs network access and a writable NLTK data directory,
    neither of which is guaranteed on a cluster. Pass your own set to
    first_word_is_real / add_first_word_real to avoid the dependency
    entirely.
    """
    import nltk
    from nltk.corpus import words

    if download:
        nltk.download("words", quiet=True)
    return {w.lower() for w in words.words()}


def first_word_is_real(text, vocabulary: set) -> int:
    """1 when the leading token looks like an English word, else 0.

    Three ways to score 0, and each is deliberate:
      * two characters or fewer -- too short to be a meaningful noun, and
        usually a code fragment;
      * contains a digit -- part numbers and sizes;
      * not in the vocabulary -- vendor codes, abbreviations, model names.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    first_word = text.split(maxsplit=1)[0]
    first_clean = first_word.strip('.,;:!?"\'()[]{}').lower()
    if len(first_clean) <= 2:
        return 0
    if any(char.isdigit() for char in first_clean):
        return 0
    return 1 if first_clean in vocabulary else 0


def add_first_word_real(
    df: pd.DataFrame, column: str, vocabulary: set, out_col: str = FIRST_WORD_COL
) -> pd.DataFrame:
    """Add the 0/1 flag as a column, keeping it for later inspection."""
    out = df.copy()
    out[out_col] = out[column].apply(lambda t: first_word_is_real(t, vocabulary))
    return out


def strip_first_term(
    df: pd.DataFrame, column: str, flag_col: str = FIRST_WORD_COL
) -> pd.DataFrame:
    """Drop the leading token wherever the flag says it is not a real word.

    Only the first token, and only once -- a description opening with two
    codes keeps the second. That is the original behaviour; whether it
    should recurse is a real question, and one worth answering with the
    counts rather than by assumption.
    """
    out = df.copy()

    def _strip(row):
        if row[flag_col] == 0 and isinstance(row[column], str):
            parts = row[column].split(maxsplit=1)
            return parts[1] if len(parts) > 1 else ""
        return row[column]

    out[column] = out.apply(_strip, axis=1)
    return out


# ---------------------------------------------------------------------------
# The regex procedure
# ---------------------------------------------------------------------------

def build_patterns(lists: Dict[str, List[str]]) -> Dict[str, object]:
    """The four pattern groups, built from the reference lists.

    Returned rather than applied so they can be printed and inspected --
    a wall of regex that only ever runs is a wall of regex nobody checks.

    Note the lists are interpolated raw, not regex-escaped, except for
    `sizes` in the SZ pattern. That is how the original was written; terms
    containing regex metacharacters would therefore behave as patterns
    rather than literals.
    """
    colors = lists["colors"]
    color_abbrevs = lists["color_abbrevs"]
    sizes = lists["sizes"]
    units = lists["units"]
    unit_abbrevs = lists["unit_abbrevs"]
    unit_plurals = lists["unit_plurals"]

    spec_units_pattern = "|".join(SPEC_UNITS)
    non_spec_token = r"\d+|\.|/|-|X"
    all_tokens = rf"{non_spec_token}|{spec_units_pattern}"
    sizes_pattern = "|".join(re.escape(s) for s in sizes)

    deletion_1 = [
        # DV or DC indicators
        r"\(DV\)|\[DV\]|\(DC\)|\[DC\]",
        # number followed by a slash and then PK, etc.
        r"(?<=\s)\d+/(CT|CNT|CA|PK|BX|BG|CASE|PACK|BOX|BAG|RL|ROLL)",
        # number (or number followed by a dash) followed by CNT/PK, etc.
        r"(?<=\s)\d+(?:-)?/(CNT|CT)/(CA|PK|BX|BG|RL|ROLL)",
        # combos of digits, "X", units like FT/IN, dashes and slashes
        rf"(?<=\s)(?:{non_spec_token})(?:{all_tokens}){{1,}}(?=\s|$)",
        # periods
        r"[.]",
        r"(?:\d+-)?\d+/\d+[Xx]\d+",
        r"\d+/\d+[Xx]\d+-\d+/\d+",
        r"\d+[Xx]\d+-\d+/\d+",
        # #/#X#
        r"\d+/\d+[Xx]\d+",
        r"\d+-\d+/\d+",
        # #/#
        r"\d+/\d+",
    ]

    replacement_1 = {
        # replace apostrophe with a space
        r"[']": " ",
    }

    deletion_2 = [
        # delete #X#<unit> or #X#X#<unit>
        r"(?<=\s)\d+(?:(?:X\d+)|(" + "|".join(unit_abbrevs) + r"))*("
        + "|".join(unit_abbrevs) + r")(?=\s|$)",
        # numbers followed by unit abbreviation, if followed by space or end
        r"(?<=\s)\d+(" + "|".join(unit_abbrevs) + r")(?=\s|$)",
        # PSI, HP, or a number followed by PSI or HP
        r"(?<=\s)(?:\d+\s*(?:PSI|HP)|PSI|HP)",
        # color abbrevs
        r"(?<=\s)(" + "|".join(color_abbrevs) + r")(?=\s|$)",
        # colors
        r"(?<=\s)(" + "|".join(colors) + r")(?=\s|$)",
        # sizes
        r"(?:^|\s)(" + "|".join(sizes) + r")/(" + "|".join(sizes) + r")(?=\s|$)",
        r"(?<=\s)(" + "|".join(sizes) + r")(?=\s|$)",
        # unit plurals
        r"(?<=\s)(" + "|".join(unit_plurals) + r")(?=\s|$)",
        # units
        r"(?<=\s)(" + "|".join(units) + r")(?=\s|$)",
        # unit abbrevs
        r"(?<=\s)(" + "|".join(unit_abbrevs) + r")(?=\s|$)",
        # numbers followed by CT, CNT, CA, etc.
        r"(?<=\s)\d+(CT|CNT|CA|PK|BX|BG)(?=\s|$)",
        # instances of "W/"
        r"W/",
        # instances of #X# mixed with slashes and/or dashes
        r"(?i)(?<=\s)(?:(?:\d+(?:\.\d+)?|X|/|-)){2,}(?=\s|$)",
        # things like 1-1/4X3-1/2
        r"\d+-\d+/\d+\s*X\s*\d+-\d+/\d+",
        # things like 1-1/2
        r"\b\d+-\d+/\d+",
        # special chars
        r"[-\./()<>#&+\"!@$%^\*]",
        # instances of #X#
        r"\d+X\d+",
        # numbers preceded by space and followed by space or end of text
        r"(?<=\s)\d+(?=\s|$)",
        # numbers followed by "T" (inclusive)
        r"(?<=\s)\d+T(?=\s|$)",
        # stray instances of "X"
        r"(?<=\s)X(?=\s|$)",
        # stray instances of "SZ", or "SZ" followed by numbers
        rf"SZ(?=\s|$)|SZ\d\S*|\d+SZ\S*|SZ[./\-:;]\S*|SZ(?:{sizes_pattern})\b",
    ]

    replacement_2 = {
        # replace multiple consecutive spaces with a single space
        r"\s{2,}": " ",
    }

    return {
        "deletion_1": deletion_1,
        "replacement_1": replacement_1,
        "deletion_2": deletion_2,
        "replacement_2": replacement_2,
    }


def apply_patterns(series: pd.Series, patterns: Dict[str, object]) -> pd.Series:
    """The two-phase pass on a bare Series.

    Training and scoring both come through here, which is the point: the
    vectorizer's vocabulary was fitted on the output of these patterns, so
    a second implementation for scoring would be a second chance to drift.
    """
    out = series.astype(str)
    for pattern in patterns["deletion_1"]:
        out = out.str.replace(pattern, "", regex=True)
    for pattern, replacement in patterns["replacement_1"].items():
        out = out.str.replace(pattern, replacement, regex=True)
    for pattern in patterns["deletion_2"]:
        out = out.str.replace(pattern, "", regex=True)
    for pattern, replacement in patterns["replacement_2"].items():
        out = out.str.replace(pattern, replacement, regex=True)
    return out


def build_expansion_pattern(expansions: Dict[str, str]) -> Optional[re.Pattern]:
    """One alternation matching any abbreviation, longest first.

    Longest first for the same reason sort_by_length exists: without it a
    short key can match inside a longer one and the longer entry never
    fires.
    """
    if not expansions:
        return None
    keys = sorted(expansions, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


def expand_abbreviations(series: pd.Series, expansions: Dict[str, str]) -> pd.Series:
    """Replace whole-word abbreviations with their expansions.

    One pass, not one pass per entry. Replacing in sequence would let an
    expansion be re-matched by a later abbreviation -- expand SS to
    STAINLESS STEEL, then have a separate ST entry rewrite part of it. A
    single alternation consumes each match once and moves on, so the
    output depends only on the file, not on the order of its rows.

    Runs last in the procedure, on text that has already had its
    punctuation removed, so \\b sits between plain alphanumerics.
    """
    pattern = build_expansion_pattern(expansions)
    if pattern is None:
        return series
    return series.astype(str).str.replace(
        pattern, lambda m: expansions[m.group(0)], regex=True
    )


def regex_cleaning_proc(
    df: pd.DataFrame,
    column: str,
    lists: Optional[Dict[str, List[str]]] = None,
    patterns: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """The two-phase pass: compounds, then reference terms and punctuation."""
    if patterns is None:
        if lists is None:
            lists = {k: upper(v) for k, v in load_cleaning_lists().items()}
        patterns = build_patterns(lists)

    out = df.copy()
    out[column] = apply_patterns(out[column], patterns)
    return out


# ---------------------------------------------------------------------------
# The recipe as one object
# ---------------------------------------------------------------------------

@dataclass
class DescriptionCleaner:
    """The whole cleaning recipe, in one picklable object.

    Exists so that training and scoring cannot use different recipes. The
    vectorizer's vocabulary is fitted on whatever comes out of here, so a
    scoring path that cleans even slightly differently sees mostly
    out-of-vocabulary words -- and fails silently, as a raised OOV rate
    rather than an error.

    Save it with the model (artifacts.save_bundle takes cleaning_cfg=) and
    load it back at scoring time. That includes the English vocabulary, so
    scoring needs no NLTK download and cannot pick up a different word
    list than training used.
    """

    lists: Dict[str, List[str]]
    vocabulary: set = field(default_factory=set)
    strip_leading_code: bool = True
    uppercase: bool = False
    patterns: Optional[Dict[str, object]] = None
    # Abbreviation expansion, applied after everything else. The mapping
    # travels with the cleaner so scoring expands exactly as training did,
    # even if the CSV on disk changes afterwards.
    expand_abbrevs: bool = False
    expansions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.patterns is None:
            self.patterns = build_patterns(self.lists)

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Clean text without dropping anything.

        Row-preserving on purpose: scoring needs one output per input, and
        a description reduced to nothing is a decision for the readability
        gate rather than something to silently discard here.
        """
        out = series.fillna("").astype(str)
        if self.uppercase:
            out = out.str.upper()
        if self.strip_leading_code:
            out = out.map(self._strip_leading)
        out = apply_patterns(out, self.patterns)
        out = out.str.strip()
        # Last, deliberately: the earlier phases delete units, sizes and
        # punctuation, so an abbreviation only has its final surroundings
        # once they have run.
        if self.expand_abbrevs and self.expansions:
            out = expand_abbreviations(out, self.expansions)
        return out.str.strip()

    def _strip_leading(self, text: str) -> str:
        if first_word_is_real(text, self.vocabulary):
            return text
        parts = text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def __call__(self, series: pd.Series) -> pd.Series:
        return self.clean_series(series)

    def describe(self) -> str:
        return (
            f"DescriptionCleaner(lists={sorted(self.lists)}, "
            f"vocabulary={len(self.vocabulary):,} words, "
            f"strip_leading_code={self.strip_leading_code}, "
            f"uppercase={self.uppercase}, "
            f"expand_abbrevs={self.expand_abbrevs}"
            + (f" ({len(self.expansions):,} pairs)" if self.expand_abbrevs else "")
            + ")"
        )


def build_cleaner(
    lists: Optional[Dict[str, List[str]]] = None,
    vocabulary: Optional[set] = None,
    strip_leading_code: bool = True,
    uppercase: bool = False,
    expand_abbrevs: bool = False,
    expansions: Optional[Dict[str, str]] = None,
) -> DescriptionCleaner:
    """A DescriptionCleaner with the packaged lists and NLTK vocabulary.

    expand_abbrevs reads cleaning_lists/abbrevs_to_expand.csv. A missing
    file raises when expansion is asked for, rather than quietly skipping
    the step.
    """
    if lists is None:
        lists = {k: upper(v) for k, v in load_cleaning_lists().items()}
    if vocabulary is None and strip_leading_code:
        vocabulary = load_english_vocabulary()
    if expansions is None:
        expansions = load_expansions(required=expand_abbrevs) if expand_abbrevs else {}
    return DescriptionCleaner(
        lists=lists,
        vocabulary=vocabulary or set(),
        strip_leading_code=strip_leading_code,
        uppercase=uppercase,
        expand_abbrevs=expand_abbrevs,
        expansions=expansions,
    )


# ---------------------------------------------------------------------------
# The whole procedure
# ---------------------------------------------------------------------------

def clean_descriptions(
    df: pd.DataFrame,
    column: str = "FullDesc",
    lists: Optional[Dict[str, List[str]]] = None,
    vocabulary: Optional[set] = None,
    orig_col: Optional[str] = DEFAULT_ORIG_COL,
    strip_leading_code: bool = True,
    uppercase: bool = False,
    expand_abbrevs: bool = False,
    expansions: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Preserve the original, drop blanks, strip leading codes, run the regex.

    Returns a new frame. Row counts are reported at each step because the
    steps drop rows, and a total that shifts without explanation is the
    thing most likely to go unnoticed here.

    uppercase is off to match the original procedure, which assumes the
    ERP descriptions arrive uppercase. Switch it on if that stops being
    true -- the patterns and reference lists are uppercase, so lowercase
    input would slip past almost all of them.
    """
    out = df.copy()
    steps = [("loaded", len(out))]

    if lists is None:
        lists = {k: upper(v) for k, v in load_cleaning_lists().items()}

    if orig_col:
        out[orig_col] = out[column]

    if uppercase:
        out[column] = out[column].astype(str).str.upper()

    out = out.dropna(subset=[column])
    out = out[out[column].astype(str).str.strip() != ""]
    steps.append(("after dropping null/blank descriptions", len(out)))

    if strip_leading_code:
        if vocabulary is None:
            vocabulary = load_english_vocabulary()
        out = add_first_word_real(out, column, vocabulary)
        n_stripped = int((out[FIRST_WORD_COL] == 0).sum())
        out = strip_first_term(out, column)
        steps.append((f"after stripping {n_stripped:,} leading codes", len(out)))

    out = regex_cleaning_proc(out, column, lists)
    out[column] = out[column].astype(str).str.strip()

    out = out[out[column] != ""]
    steps.append(("after the regex pass, blanks removed", len(out)))

    # Last step in the procedure. It substitutes rather than deletes, so
    # it cannot empty a row and the count below cannot fall.
    if expand_abbrevs:
        if expansions is None:
            expansions = load_expansions(required=True)
        out[column] = expand_abbreviations(out[column], expansions)
        steps.append((f"after expanding {len(expansions):,} abbreviation(s)", len(out)))

    out = out.reset_index(drop=True)
    if verbose:
        for label, n in steps:
            print(f"[clean] {n:>9,}  {label}")
        dropped = steps[0][1] - steps[-1][1]
        print(f"[clean] {dropped:>9,}  rows dropped in total "
              f"({dropped / steps[0][1]:.2%})" if steps[0][1] else "")
    out.attrs["steps"] = steps
    return out
