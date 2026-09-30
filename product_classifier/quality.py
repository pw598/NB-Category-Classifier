"""Input-quality signals, and the guard that uses them.

Confidence answers "how sure is the model?". It does not answer "did the
model have anything to work with?", and those come apart badly on short
descriptions. A two-word description containing no word the model has
ever seen still produces a confident-looking answer -- one derived
entirely from how common each category is, not from the product.

Measured on the current golden set, held-out descriptions run about 4.5%
out-of-vocabulary, and roughly 0.6% contain no usable word at all. Those
last ones should never be auto-accepted whatever their score says, and
the check costs nothing: it needs no labels, so it works identically in
production and on test data.

The same numbers double as a drift alarm. The relationship between
confidence and accuracy is learned once from labelled data, and holds
only while new products resemble the old ones. Out-of-vocabulary rate is
the cheapest available proxy for "these products no longer look like the
ones we calibrated on", and unlike accuracy it can be watched
continuously without anyone checking answers by hand.

Nothing here is imported by the rest of the package.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Measuring the input
# ---------------------------------------------------------------------------

def describe_inputs(clf, texts: pd.Series) -> pd.DataFrame:
    """Per-row description quality, judged against the model's vocabulary.

    Uses the fitted vectorizer's own tokeniser, so 'known' means exactly
    what the model means by it -- after the same lowercasing, token
    pattern, stop-word list and min_df pruning that training applied.

    Columns:
        n_tokens        words found in the description
        n_known         how many of those the model has a weight for
        n_unknown       the rest, which the model silently ignores
        oov_rate        n_unknown / n_tokens (0.0 for an empty description)
        usable          whether the model had anything at all to go on
    """
    if getattr(clf, "vectorizer", None) is None:
        raise RuntimeError("Classifier has no fitted vectorizer.")
    analyzer = clf.vectorizer.build_analyzer()
    vocab = clf.vectorizer.vocabulary_

    texts = texts.fillna("").astype(str)
    n_tokens = np.empty(len(texts), dtype=np.int32)
    n_known = np.empty(len(texts), dtype=np.int32)

    for i, t in enumerate(texts):
        toks = analyzer(t)
        n_tokens[i] = len(toks)
        n_known[i] = sum(1 for w in toks if w in vocab)

    n_unknown = n_tokens - n_known
    with np.errstate(invalid="ignore", divide="ignore"):
        oov = np.where(n_tokens > 0, n_unknown / np.maximum(n_tokens, 1), 0.0)

    return pd.DataFrame(
        {
            "n_tokens": n_tokens,
            "n_known": n_known,
            "n_unknown": n_unknown,
            "oov_rate": oov,
            "usable": n_known > 0,
        }
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

def apply_input_guard(
    chosen: pd.DataFrame,
    quality: pd.DataFrame,
    level_columns: List[str],
    min_known_tokens: int = 1,
    max_oov_rate: float = 1.0,
) -> pd.DataFrame:
    """Force depth 0 on rows the model had no real evidence for.

    Applied *after* a depth policy has run, and it only ever makes the
    answer shallower -- it can move a row to 'hand over entirely', never
    deepen one.

    min_known_tokens: a row with fewer recognised words than this is
        handed over. The default of 1 catches only the genuinely empty
        case -- descriptions where every single word is unknown. Raising
        it to 2 is defensible given a median description of about four
        words, but it will abstain on considerably more rows, so check
        the reported count before adopting it.

    max_oov_rate: hand over rows where more than this share of words are
        unrecognised, regardless of how many are known. Off by default.

    Returns a copy with 'Guard Triggered' added.
    """
    if len(chosen) != len(quality):
        raise ValueError(
            f"chosen has {len(chosen)} rows, quality has {len(quality)}; "
            "they must describe the same items in the same order."
        )

    q = quality.reset_index(drop=True)
    out = chosen.reset_index(drop=True).copy()

    triggered = (q["n_known"].to_numpy() < min_known_tokens) | (
        q["oov_rate"].to_numpy() > max_oov_rate
    )
    triggered &= out["Assigned Depth"].to_numpy() > 0  # already handed over

    out.loc[triggered, "Assigned Depth"] = 0
    for c in level_columns:
        if c in out.columns:
            out.loc[triggered, c] = ""
    out["Guard Triggered"] = triggered

    n = int(triggered.sum())
    print(
        f"[guard] {n:,} of {len(out):,} rows ({n / max(len(out), 1):.3%}) "
        f"handed over for lack of usable words"
    )
    return out


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def baseline_profile(quality: pd.DataFrame) -> Dict[str, float]:
    """Reference figures from the data the thresholds were calibrated on.

    Store this alongside the model. It is what future batches get
    compared against.
    """
    return {
        "n_rows": float(len(quality)),
        "mean_oov_rate": float(quality["oov_rate"].mean()),
        "median_tokens": float(quality["n_tokens"].median()),
        "share_unusable": float((~quality["usable"]).mean()),
        "share_under_3_known": float((quality["n_known"] < 3).mean()),
    }


def drift_report(
    quality: pd.DataFrame,
    baseline: Dict[str, float],
    oov_tolerance: float = 1.5,
) -> pd.DataFrame:
    """Compare a new batch against the calibration baseline.

    oov_tolerance is a multiplier: at the default, an out-of-vocabulary
    rate more than 1.5x the baseline is flagged. A flag does not mean the
    predictions are wrong -- it means the accuracy promised by the
    thresholds was measured on different-looking products and may no
    longer apply, so the thresholds are due a refresh against freshly
    labelled examples.
    """
    current = baseline_profile(quality)
    rows = []
    for k, base in baseline.items():
        if k == "n_rows":
            continue
        now = current[k]
        limit = base * oov_tolerance if base else np.inf
        rows.append(
            {
                "measure": k,
                "baseline": base,
                "current": now,
                "ratio": now / base if base else np.nan,
                "flagged": bool(now > limit),
            }
        )
    report = pd.DataFrame(rows)
    if report["flagged"].any():
        print(
            "[drift] "
            + ", ".join(report.loc[report["flagged"], "measure"])
            + " above tolerance -- recalibrate before trusting the thresholds."
        )
    else:
        print("[drift] batch resembles the calibration data.")
    return report
