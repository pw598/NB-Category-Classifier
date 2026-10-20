"""cleaning_method_20261015 must reproduce cleaning_repo/01_clean_data.ipynb exactly.

tests/data/cleaning_method_20261015_golden.json holds descriptions (a sample of
sample_spend_data.txt plus hand-written cases) and the output of the original
cleaning_repo code with the notebook's settings, using a fixed English word
list so the test needs no NLTK download.
"""

import copy
import json
import pickle
from pathlib import Path

import pandas as pd

from nbclassifier.cleaning import available_methods, build_cleaner

METHOD = "cleaning_method_20261015"
GOLDEN = json.loads((Path(__file__).parent / "data" / f"{METHOD}_golden.json").read_text())


def _cleaner(**kw):
    return build_cleaner(METHOD, english=set(GOLDEN["english"]), verbose=False, **kw)


def test_method_is_available():
    assert METHOD in available_methods()


def test_matches_cleaning_repo():
    out, report = _cleaner().clean_text(pd.Series(GOLDEN["inputs"]), fit=True)
    assert list(out) == GOLDEN["expected_min5"]
    assert report["vocabulary_size"] > 0


def test_matches_cleaning_repo_min_word_1():
    c = _cleaner()
    c.settings = {**c.settings, "min_word_descriptions": 1}
    out, _ = c.clean_text(pd.Series(GOLDEN["inputs"]), fit=True)
    assert list(out) == GOLDEN["expected_min1"]


def test_vocabulary_is_frozen_after_fit():
    c = _cleaner()
    c.clean_text(pd.Series(GOLDEN["inputs"]), fit=True)
    vocab = dict(c.vocabulary)
    # cleaning new data re-uses the training vocabulary (cleaning_repo's BUILD_VOCABULARY = False)
    out, _ = c.clean_text(pd.Series(["GLOVE NITRILE ZZZUNSEENWORD"]), fit=False)
    assert c.vocabulary == vocab
    assert "ZZZUNSEENWORD" not in out[0]


def test_pickle_roundtrip_keeps_frozen_code():
    c = _cleaner()
    c.clean_text(pd.Series(GOLDEN["inputs"]), fit=True)
    c2 = pickle.loads(pickle.dumps(c))
    a, _ = c.clean_text(pd.Series(GOLDEN["inputs"]), fit=False)
    b, _ = c2.clean_text(pd.Series(GOLDEN["inputs"]), fit=False)
    assert list(a) == list(b)


def test_append_fields():
    c = _cleaner(append_fields={"Global Vendno": "VEND"}, append_nulls_as_none=True)
    c.settings = {**c.settings, "min_word_descriptions": 1}
    df = pd.DataFrame({"FullDesc": ["BALL VALVE", "GLOVE", ""], "Global Vendno": ["FREUD Inc.", None, "X"]})
    out, _ = c.clean_frame(df, "FullDesc", fit=True)
    assert list(out) == ["BALL VALVE VEND_FREUD_INC", "GLOVE VEND_None", ""]
    c2 = copy.copy(c)
    c2.append_nulls_as_none = False
    out2, _ = c2.clean_frame(df, "FullDesc", fit=False)
    assert list(out2) == ["BALL VALVE VEND_FREUD_INC", "GLOVE", ""]


def test_append_fields_with_levels():
    c = _cleaner(append_fields={"UNSPSC": {"prefix": "UNSPSC", "levels": [2, 4, 6, 8]}, "Prodline": "PLINE"},
                 append_nulls_as_none=True)
    c.settings = {**c.settings, "min_word_descriptions": 1}
    df = pd.DataFrame({"FullDesc": ["BALL VALVE", "GLOVE"], "UNSPSC": ["46181504", "4618"],
                       "Prodline": [None, "ABC"]})
    out, _ = c.clean_frame(df, "FullDesc", fit=True)
    assert list(out) == ["BALL VALVE UNSPSC2_46 UNSPSC4_4618 UNSPSC6_461815 UNSPSC8_46181504 PLINE_None",
                         "GLOVE UNSPSC2_46 UNSPSC4_4618 PLINE_ABC"]


def test_append_fields_validation():
    for bad in ({"UNSPSC": {"prefix": "UN-SPSC"}}, {"UNSPSC": {"prefix": "U", "levels": [0]}},
                {"UNSPSC": {"prefix": "U", "level": [2]}}):
        try:
            _cleaner(append_fields=bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been rejected")
