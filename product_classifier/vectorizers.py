"""Vectorizers that treat a bigram's two terms as a set, not a sequence.

Product descriptions are keyword lists, not prose. "BALL VALVE BRASS" and
"BRASS BALL VALVE" name the same thing, and the cleaning procedure makes
this worse rather than better: stripping units, sizes and colours pulls
words together that were not adjacent before, so which two terms end up
side by side is partly an accident of what was removed.

With ordered bigrams those two descriptions share only their unigrams --
the bigram evidence splits across "ball valve"/"valve brass" in one and
"brass ball"/"ball valve" in the other. Folding each bigram to its sorted
form merges them into one feature, so the pair counts as the same evidence
wherever it appears.

Only two-term n-grams are folded. Unigrams pass through untouched, and so
do trigrams and longer, which keep their order.

Implementation note: these override the public `build_analyzer`, and the
analyzer it returns is only ever a local inside sklearn's fit/transform --
it is never stored on the instance. So a fitted vectorizer pickles
normally, which matters because one travels in every saved model bundle.
The classes are module-level for the same reason: unpickling needs to
import them by name.
"""

from __future__ import annotations

from typing import Callable, List

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

__all__ = [
    "fold_bigrams",
    "UnorderedBigramCountVectorizer",
    "UnorderedBigramTfidfVectorizer",
    "build_unordered",
]


def fold_bigrams(tokens: List[str]) -> List[str]:
    """Sort the two terms of every bigram, leaving everything else alone.

    sklearn emits n-grams as space-joined strings, so a bigram is a token
    that splits into exactly two parts. The default token pattern cannot
    produce a unigram containing a space, so the split is unambiguous.
    """
    out = []
    for tok in tokens:
        parts = tok.split(" ")
        if len(parts) == 2:
            a, b = parts
            out.append(f"{a} {b}" if a <= b else f"{b} {a}")
        else:
            out.append(tok)
    return out


class UnorderedBigramCountVectorizer(CountVectorizer):
    """CountVectorizer whose bigrams ignore term order."""

    def build_analyzer(self) -> Callable[[str], List[str]]:
        base = super().build_analyzer()
        return lambda doc: fold_bigrams(base(doc))


class UnorderedBigramTfidfVectorizer(TfidfVectorizer):
    """TfidfVectorizer whose bigrams ignore term order."""

    def build_analyzer(self) -> Callable[[str], List[str]]:
        base = super().build_analyzer()
        return lambda doc: fold_bigrams(base(doc))


def build_unordered(kind: str, **params):
    """The order-insensitive counterpart of CountVectorizer/TfidfVectorizer."""
    if kind == "count":
        return UnorderedBigramCountVectorizer(**params)
    if kind == "tfidf":
        return UnorderedBigramTfidfVectorizer(**params)
    raise ValueError(f"Unknown vectorizer kind: {kind!r}. Use 'count' or 'tfidf'.")
