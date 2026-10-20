"""Vectorizer construction, including the order-insensitive bigram option.

ordered_bigrams=False folds every bigram to its sorted form, so 'GLOVE NITRILE'
and 'NITRILE GLOVE' become one feature. Unigrams and longer n-grams are left
alone. The subclasses live at module level so fitted vectorizers pickle.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .config import TOKEN_PATTERNS, HyperParams


def fold_bigrams(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        parts = t.split(" ")
        if len(parts) == 2:
            a, b = parts
            out.append(f"{a} {b}" if a <= b else f"{b} {a}")
        else:
            out.append(t)
    return out


class UnorderedBigramCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        base = super().build_analyzer()
        return lambda doc: fold_bigrams(base(doc))


class UnorderedBigramTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        base = super().build_analyzer()
        return lambda doc: fold_bigrams(base(doc))


def build_vectorizer(hp: HyperParams):
    common = dict(
        max_features=hp.max_features,
        ngram_range=tuple(hp.ngram_range),
        min_df=hp.min_df,
        max_df=hp.max_df,
        stop_words=hp.stop_words,
        binary=hp.binary,
        lowercase=True,
        token_pattern=TOKEN_PATTERNS.get(hp.token_pattern, hp.token_pattern),
        dtype=np.float64,
    )
    if hp.vectorizer_kind == "count":
        cls = CountVectorizer if hp.ordered_bigrams else UnorderedBigramCountVectorizer
        return cls(**common)
    cls = TfidfVectorizer if hp.ordered_bigrams else UnorderedBigramTfidfVectorizer
    return cls(sublinear_tf=hp.sublinear_tf, norm=hp.norm, use_idf=hp.use_idf, **common)
