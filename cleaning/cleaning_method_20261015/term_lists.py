"""Reading the term lists, and building the patterns the cleaning steps match against."""
import re
from pathlib import Path

# A number as it appears in descriptions: 12, 1.5, .070, 1,000, 1/2, 1-1/2
NUM = r"(?:\d+(?:[.,]\d+)?|\.\d+)(?:-\d+/\d+|/\d+)?"
# A size expression: a number, a thread (5/8-11, #10-32, M10X1.5), or dimensions (2X4, 3/8 X 3/4, 2X1-1/4)
SIZE_EXPR = rf"(?:#\d+-\d+|M\d+(?:\.\d+)?(?:\s*X\s*{NUM})+|{NUM}(?:-\d+)?(?:\s*X\s*{NUM}(?:-\d+)?)*)"
# Where a size expression may start: not inside a word, a number or a placeholder
SIZE_START = r"(?<![A-Z0-9<>.,/#-])"


def read_terms(path):
    """One term per line (.txt), or the first field of each row (.csv). Upper-cased; blank lines and repeats dropped.
    A heading line does no harm: it only matches a word with the heading's name."""
    path = Path(path)
    terms = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        term = line.split(",")[0] if path.suffix.lower() == ".csv" else line
        term = " ".join(term.strip().strip('"').upper().split())
        if term and term not in terms:
            terms.append(term)
    return terms


def read_units(path):
    """Parse units.txt. Returns a list of units: {"abbrev", "prefix", "forms": [(form, number_only), ...]}."""
    units = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}, line {n}: expected 'ABBREVIATION = form, form', got {line!r}")
        head, rest = line.split("=", 1)
        prefix = head.strip().startswith(">")
        abbrev = head.strip().lstrip(">").strip().upper()
        forms = []
        for form in rest.split(","):
            form = " ".join(form.strip().upper().split())
            if form:
                forms.append((form.lstrip("~").strip(), form.startswith("~")))
        units.append({"abbrev": abbrev, "prefix": prefix, "forms": forms})
    return units


def alternation(terms):
    """Regex alternation of literal terms, longest first; a space in a term matches any run of spaces."""
    return "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in sorted(set(terms), key=len, reverse=True))


def whole_words(terms):
    """Pattern matching any of the terms as a whole word. Punctuation counts as a word boundary, except that a
    single-letter term next to - . ' or / is left alone (L-SHAPED, S.S., O'RING). Placeholders are skipped."""
    long_terms = [t for t in terms if len(t) > 1]
    short_terms = [t for t in terms if len(t) == 1]
    parts = []
    if long_terms:
        parts.append(rf"(?<![A-Z0-9<>])(?:{alternation(long_terms)})(?![A-Z0-9<>])")
    if short_terms:
        parts.append(rf"(?<![A-Z0-9<>\-.'/])(?:{alternation(short_terms)})(?![A-Z0-9<>\-.'/])")
    return re.compile("|".join(parts) if parts else r"(?!x)x")


def unit_patterns(units):
    """Patterns and lookups for steps 7 to 9, built from read_units()."""
    to_abbrev = {}                       # every form, and every abbreviation, -> its abbreviation
    anywhere = []                        # forms replaced wherever they stand
    for u in units:
        to_abbrev[u["abbrev"]] = u["abbrev"]
        for form, number_only in u["forms"]:
            to_abbrev[form] = u["abbrev"]
            if not number_only:
                anywhere.append(form)
    abbrevs = sorted({u["abbrev"] for u in units})
    prefixes = sorted({u["abbrev"] for u in units if u["prefix"]})
    word_forms = [f for f in to_abbrev if f[-1].isalnum()]
    mark_forms = [f for f in to_abbrev if not f[-1].isalnum()]        # '"', "'", "#", "°" after a number
    unit_after_number = rf"(?:(?:{alternation(word_forms)})(?![A-Z0-9<>])" + (f"|{alternation(mark_forms)})" if mark_forms else ")")
    return {
        "to_abbrev": to_abbrev,
        "anywhere": whole_words(anywhere),
        # step 7: a number, an optional space, hyphen or slash (100/PK), then any form of a unit
        "after_number": re.compile(rf"{SIZE_START}(?P<num>{SIZE_EXPR})(?P<sep>[\s/-]*)(?P<unit>{unit_after_number})"),
        # step 8: a number, a space, hyphen or slash, then an abbreviation
        "glue": re.compile(rf"{SIZE_START}(?P<num>{SIZE_EXPR})[\s/-]+(?P<unit>{alternation(abbrevs)})(?![A-Z0-9<>])"),
        # step 8: a unit glued between two dimensions (9/16INX535FT) gets spaces around the X
        "unit_x": re.compile(rf"(?<=\d)(?P<unit>{alternation(abbrevs)})X(?=[\d.#])"),
        # step 8: a prefix unit (GR) followed by a number, with no number glued in front of it
        "glue_prefix": re.compile(rf"(?<![A-Z0-9<>.])(?P<unit>{alternation(prefixes)})[\s-]+(?P<num>\d+(?:\.\d+)?)(?![A-Z0-9<>./])"
                                  if prefixes else r"(?!x)x"),
        # step 9: a size expression, optionally glued to an abbreviation
        "size": re.compile(rf"{SIZE_START}(?P<size>{SIZE_EXPR})(?P<unit>{alternation(abbrevs)})?(?![A-Z0-9<>])"),
    }


def english_words(words_file=None):
    """Lower-cased English words: from words_file (one per line) if given, else the NLTK word list (downloaded if missing)."""
    if words_file:
        return {w.strip().lower() for w in Path(words_file).read_text(encoding="utf-8-sig").splitlines() if w.strip()}
    import nltk
    from nltk.corpus import words
    try:
        return {w.lower() for w in words.words()}
    except LookupError:
        nltk.download("words", quiet=True)
        return {w.lower() for w in words.words()}
