"""The cleaning steps used by 01_clean_data. Each step takes a list of descriptions and returns a new list."""
import collections
import random
import re

import term_lists as tl

PLACEHOLDER = re.compile(r"<[A-Z]+>")
_PUNCT = '.,;:!?"\'()[]{}'


def show_changes(step, before, after, examples=3, seed=0):
    """Print how many descriptions a step changed, with a few examples."""
    changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    print(f"{step}: {len(changed):,} of {len(before):,} descriptions changed ({len(changed) / max(len(before), 1):.1%})")
    for i in random.Random(seed).sample(changed, min(examples, len(changed))):
        print(f"    {before[i]!r}\n -> {after[i]!r}")


# ---------------------------------------------------------------- step 3
def repair_split_words(texts, field_width=24, min_count=3):
    """Rejoin a word split by the space after character `field_width`, when the joined word occurs elsewhere at
    least `min_count` times and at least as often as the rarer of its two pieces."""
    w = field_width
    at_boundary = [len(t) > w + 1 and t[w] == " " and t[w - 1] != " " and t[w + 1] != " " for t in texts]
    count = collections.Counter()                        # whole words, leaving out the pieces at the boundary
    for t, boundary in zip(texts, at_boundary):
        parts = t.split()
        if boundary:
            k = len(t[:w].split())
            parts = parts[:k - 1] + parts[k + 1:]
        count.update(parts)
    out = list(texts)
    for i, t in enumerate(texts):
        if at_boundary[i]:
            left, right = t[:w].split()[-1], t[w + 1:].split()[0]
            joined = left + right
            if count[joined] >= min_count and count[joined] >= min(count[left], count[right]):
                out[i] = t[:w] + t[w + 1:]
    return out


# ---------------------------------------------------------------- step 4
def remove_markers(texts, markers):
    """Remove each marker wherever it appears; "*" in a marker stands for any run of characters inside it."""
    pattern = re.compile("|".join(re.escape(m.upper()).replace(r"\*", r"[^)\]]*") for m in markers))
    return [" ".join(pattern.sub(" ", t).split()) if pattern.search(t) else t for t in texts]


# ---------------------------------------------------------------- step 5
def first_word_is_real(word, vocabulary):
    """Your rule: a real word has more than two characters, no digits, and is in the English word list."""
    clean = word.strip(_PUNCT).lower()
    return len(clean) > 2 and not any(ch.isdigit() for ch in clean) and clean in vocabulary


def _in_vocabulary(w, vocabulary):
    """w, or its singular, is an English word."""
    return (w in vocabulary or (w.endswith("ies") and w[:-3] + "y" in vocabulary)
            or (w.endswith("es") and w[:-2] in vocabulary) or (w.endswith("s") and w[:-1] in vocabulary))


def is_word_form(word, vocabulary):
    """A plural, hyphenated or run-together form of English words: GOGGLES, O-RING, LOCKNUT."""
    t = word.strip(_PUNCT).lower()
    if len(t) <= 2 or not t.replace("-", "").isalpha():
        return False
    if "-" in t:
        parts = [p for p in t.split("-") if len(p) >= 3]
        return bool(parts) and all(_in_vocabulary(p, vocabulary) for p in parts)
    return _in_vocabulary(t, vocabulary) or any(t[:i] in vocabulary and _in_vocabulary(t[i:], vocabulary)
                                                 for i in range(3, len(t) - 2))


def is_measurement(word, patterns):
    """A size, thread or fraction such as 7/16-20, 1/2IN or 3/8X3/4; four or more digits in a row make it a code."""
    t = word.strip(_PUNCT)
    if re.search(r"\d{4,}", t) or not any(ch.isdigit() for ch in t):
        return False
    t = standardize_units([t], patterns)[0].replace(" ", "")
    m = patterns["size"].fullmatch(t)
    return bool(m) and (m.group("unit") is not None or any(c in m.group("size") for c in "/.X#-"))


def strip_first_word(texts, vocabulary, good_tokens, patterns, keep_word_forms=True, keep_measurements=True,
                     keep_common=50):
    """Remove the first word unless it is a real English word or a good token. Safeguards keep a first word that is
    a form of English words, a measurement, or a letters-only term found in at least `keep_common` descriptions.
    Returns (texts, removed: {word: descriptions}, kept_by_safeguard: {reason: descriptions})."""
    good = {g.lower() for g in good_tokens}
    seen = collections.Counter()
    if keep_common:
        for t in texts:
            seen.update({w.strip(_PUNCT) for w in t.split()})
    out, removed, kept = [], collections.Counter(), collections.Counter()
    for t in texts:
        parts = t.split(maxsplit=1)
        if not parts or first_word_is_real(parts[0], vocabulary) or parts[0].strip(_PUNCT).lower() in good:
            out.append(t)
            continue
        word = parts[0]
        reason = ("word form" if keep_word_forms and is_word_form(word, vocabulary) else
                  "measurement" if keep_measurements and is_measurement(word, patterns) else
                  "common abbreviation" if keep_common and word.strip(_PUNCT).replace("-", "").isalpha()
                  and seen[word.strip(_PUNCT)] >= keep_common else None)
        if reason:
            kept[reason] += 1
            out.append(t)
        else:
            removed[word.strip(_PUNCT)] += 1
            out.append(parts[1] if len(parts) > 1 else "")
    return out, removed, kept


# ---------------------------------------------------------------- step 6
def replace_colors(texts, colors):
    """Replace every color and color abbreviation (whole words) with <COLOR>."""
    pattern = tl.whole_words(colors)
    return [pattern.sub("<COLOR>", t) for t in texts]


# ---------------------------------------------------------------- step 7
def standardize_units(texts, patterns):
    """Replace every form of a unit with its abbreviation: forms marked "~" only when they follow a number."""
    to_abbrev = patterns["to_abbrev"]
    key = lambda s: " ".join(s.split())
    out = []
    for t in texts:
        t = patterns["after_number"].sub(lambda m: m.group("num") + m.group("sep") + to_abbrev[key(m.group("unit"))], t)
        t = patterns["anywhere"].sub(lambda m: to_abbrev[key(m.group())], t)
        out.append(t)
    return out


# ---------------------------------------------------------------- step 8
def glue_units(texts, patterns):
    """Delete the space (or hyphen) between a number and the unit after it: 12 PC -> 12PC. Then glue a prefix unit
    to the number after it, if no number is glued in front of it: GR 8 -> GR8, but 8 GR 5 -> 8GR 5."""
    out = []
    for t in texts:
        t = patterns["unit_x"].sub(lambda m: m.group("unit") + " X ", t)
        t = patterns["glue"].sub(lambda m: m.group("num") + m.group("unit"), t)
        t = patterns["glue_prefix"].sub(lambda m: m.group("unit") + m.group("num"), t)
        out.append(t)
    return out


# ---------------------------------------------------------------- step 9
# a number standing on its own: not glued to letters, and not part of a larger number such as 1/4
_STRAY_DIGITS = re.compile(r"(?<![\w<>.,/-])\d+(?:[.,/-]\d+)*(?![\w<>.,/-])")
# a size expression glued straight onto letters (1/2-13X5-1/2ALLOY, 1/16TH): the whole expression is the size
_SIZE_GLUED = re.compile(rf"{tl.SIZE_START}(?P<size>{tl.SIZE_EXPR})(?=[A-Z])(?!X\s*[\d.#])")   # not cut at an X before more digits


def replace_sizes(texts, sizes, patterns):
    """Replace clothing sizes with <SIZE>; a number glued to a unit with <SIZE> plus the unit (1/2IN -> <SIZE>IN);
    size patterns without a unit (2X4, 1-1/4, 2X1-1/4, 5/8-11, .070) with <SIZE>. A size expression glued to other
    letters is replaced whole, keeping the letters (1/2-13X5-1/2ALLOY -> <SIZE>ALLOY, 1/16TH -> <SIZE>TH), as long
    as it contains / . X or # (so a part number such as 12-345ABC is left alone). Then delete stray digits."""
    clothing = tl.whole_words(sizes)

    def glued(m):
        return "<SIZE>" if any(c in m.group("size") for c in "/.X#") else m.group()

    def size(m):
        if m.group("unit"):
            return "<SIZE>" + m.group("unit")
        return "<SIZE>" if any(c in m.group("size") for c in "/.X#-") else m.group()

    out = []
    for t in texts:
        t = clothing.sub("<SIZE>", t)
        t = _SIZE_GLUED.sub(glued, t)
        t = patterns["size"].sub(size, t)
        t = _STRAY_DIGITS.sub(" ", t)
        out.append(" ".join(t.split()))
    return out


# ---------------------------------------------------------------- step 10
_INSIDE_WORD = re.compile(r"(?<=[^\W\d_])[.'’](?=[^\W\d_])")
_PUNCTUATION = re.compile(r"(<[A-Z]+>)|[^\w\s]|_")


def remove_punctuation(texts, mode="space"):
    """Remove punctuation; placeholders such as <SIZE> are kept.
    mode="space":  punctuation becomes a space (T-MAX -> T MAX), except . and ' between letters (S.S. -> SS).
    mode="delete": punctuation is deleted (T-MAX -> TMAX)."""
    if mode not in ("space", "delete"):
        raise ValueError('mode must be "space" or "delete"')
    fill = " " if mode == "space" else ""
    out = []
    for t in texts:
        if mode == "space":
            t = _INSIDE_WORD.sub("", t)
        out.append(_PUNCTUATION.sub(lambda m: m.group(1) or fill, t).replace("><", "> <"))
    return out


# ---------------------------------------------------------------- step 11
def build_vocabulary(texts, min_descriptions):
    """Words found in at least `min_descriptions` descriptions, with their counts, most common first.
    Words containing a placeholder are always kept."""
    count = collections.Counter()
    for t in texts:
        count.update(set(t.split()))
    return {w: n for w, n in count.most_common() if n >= min_descriptions or "<" in w}


def save_vocabulary(vocabulary, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("word\tdescriptions\n")
        f.writelines(f"{w}\t{n}\n" for w, n in vocabulary.items())


def load_vocabulary(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return {line.split("\t")[0]: int(line.split("\t")[1]) for line in f if line.strip()}


def remove_rare_words(texts, vocabulary):
    """Keep only words in the vocabulary (and any word containing a placeholder)."""
    return [" ".join(w for w in t.split() if w in vocabulary or "<" in w) for t in texts]


# ---------------------------------------------------------------- step 12
def collapse_repeats(texts, placeholders=("<SIZE>", "<COLOR>")):
    """Replace a run of the same placeholder with one: <SIZE> <SIZE> <SIZE> -> <SIZE>. Only whole words count, so
    <SIZE> <SIZE>IN is left alone."""
    alts = "|".join(re.escape(p) for p in placeholders)
    pattern = re.compile(rf"(?<!\S)({alts})(?!\S)(?:\s+\1(?!\S))+")
    return [pattern.sub(r"\1", t) for t in texts]


# ---------------------------------------------------------------- step 13
def tidy_spaces(texts):
    return [" ".join(t.split()) for t in texts]
