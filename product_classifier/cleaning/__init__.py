"""Text-cleaning utilities for product descriptions."""

from .descriptions import (
    DescriptionCleaner,
    build_cleaner,
    apply_patterns,
    clean_descriptions,
    build_patterns,
    regex_cleaning_proc,
    first_word_is_real,
    add_first_word_real,
    strip_first_term,
    load_english_vocabulary,
    expand_abbreviations,
    build_expansion_pattern,
)
from .lists import (
    EXPANSIONS_FILENAME,
    load_cleaning_lists,
    load_expansions,
    sort_by_length,
    upper,
)
from .text import (
    CleaningConfig,
    clean_dataframe,
    clean_series,
    collapse_whitespace,
    regex_deletion,
    regex_replacement,
    word_boundary_pattern,
)

__all__ = [
    "CleaningConfig",
    "DescriptionCleaner",
    "build_cleaner",
    "apply_patterns",
    "clean_descriptions",
    "build_patterns",
    "regex_cleaning_proc",
    "first_word_is_real",
    "add_first_word_real",
    "strip_first_term",
    "load_english_vocabulary",
    "expand_abbreviations",
    "build_expansion_pattern",
    "load_expansions",
    "EXPANSIONS_FILENAME",
    "clean_dataframe",
    "clean_series",
    "collapse_whitespace",
    "load_cleaning_lists",
    "regex_deletion",
    "regex_replacement",
    "sort_by_length",
    "upper",
    "word_boundary_pattern",
]
