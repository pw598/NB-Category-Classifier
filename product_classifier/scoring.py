"""Score raw descriptions the way a service would: clean, gate, predict.

One function, `score_descriptions`, doing the three things that have to
happen in order:

1. **Clean** the incoming text with the same recipe the training data went
   through. The vocabulary was fitted on cleaned descriptions, so raw text
   arriving at the model looks mostly unknown -- get this wrong and every
   score is computed on text the model cannot read.

2. **Gate on readability.** A description whose every word is unknown
   still produces a confident-looking answer, because the model falls back
   on how common each category is. Nothing about the confidence figure
   reveals that, so it has to be checked separately and refused outright.

3. **Predict at the deepest level that clears its threshold**, testing
   deepest first. Failing every threshold is a refusal too, and a
   different one from being unreadable -- an unreadable description needs
   better data, one that is merely uncertain needs a human.

Thresholds here are on the **raw, uncalibrated** scale: the product of the
per-level posteriors, summed over paths sharing the prefix. Raw scores are
not comparable between levels -- a depth-1 score and a depth-4 score have
different distributions -- so each level needs its own threshold, and a
number that looks small may still be a high bar. Take them from a
threshold-fitting run rather than intuition.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .model import HierarchicalPathClassifier

PREFIX_PROB_FMT = "L{d} Prefix Probability"
PRED_FMT = "L{d} Predicted {level}"

REFUSED_UNREADABLE = "refused: no recognisable words"
REFUSED_UNCERTAIN = "refused: below every threshold"

REPORT_MODES = ("raw", "percentile", "threshold_ratio", "model_normalised")


def _reported_confidence(
    raw: np.ndarray,
    depth: np.ndarray,
    thresholds: Dict[int, float],
    mode: str,
    reference: Optional[pd.DataFrame] = None,
    normalised: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Turn the raw decision score into something readable.

    Every mode here is applied *after* the decision has been made, so none
    of them can change what was predicted. They differ in whether they can
    visibly contradict it:

    raw
        the decision value itself. Honest, unreadable -- not comparable
        between levels, and a small number may be a high bar.

    percentile
        where this score sits among out-of-fold scores at the same depth.
        Monotone in the raw score, so it can never disagree with the
        decision, and people read "higher than 92% of products" correctly
        without knowing anything about the scale. Needs a reference table.

        One wrinkle: where many products share the same raw score, this
        reports the *top* of that tie group, which flatters a tied score
        slightly. Real raw scores are near-continuous so ties are rare,
        but a heavily-tied distribution -- a tiny vocabulary, say -- will
        show it.

    threshold_ratio
        score / threshold at the assigned depth. 1.0 means exactly at the
        bar, 1.4 means 40% above it. Monotone within a level, needs
        nothing stored, and directly answers "how close was this?".

    model_normalised
        the model's own normalised belief -- raw divided by the total over
        candidate paths. This is the only mode that can contradict the
        decision: normalising divides each row by its own total, so it
        reorders rows, and a refused product can show a higher figure than
        an accepted one. Offered because it is what people ask for, but
        prefer percentile if the number will be shown next to a decision.
    """
    out = np.full(len(raw), np.nan)
    scored = depth > 0
    if not scored.any():
        return out

    if mode == "raw":
        out[scored] = raw[scored]
    elif mode == "model_normalised":
        if normalised is None:
            raise ValueError("model_normalised needs the normalised scores.")
        out[scored] = normalised[scored]
    elif mode == "threshold_ratio":
        t = np.array([thresholds.get(int(d), np.nan) for d in depth], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[scored] = raw[scored] / t[scored]
    elif mode == "percentile":
        if reference is None:
            raise ValueError(
                "percentile reporting needs a confidence reference; build one "
                "with artifacts.build_confidence_reference() from out-of-fold "
                "predictions and save it in the bundle."
            )
        for d in sorted(set(depth[scored])):
            ref = reference.loc[reference["depth"] == d].sort_values("raw_score")
            if ref.empty:
                continue
            rows = depth == d
            out[rows] = np.interp(
                raw[rows], ref["raw_score"].to_numpy(), ref["quantile"].to_numpy()
            )
    else:
        raise ValueError(f"Unknown report mode {mode!r}; use one of {REPORT_MODES}.")
    return out


def score_descriptions(
    clf: HierarchicalPathClassifier,
    raw_texts,
    thresholds: Dict[int, float],
    cleaning_cfg=None,
    level_columns: Optional[List[str]] = None,
    min_known_tokens: int = 1,
    lookup: Optional[pd.DataFrame] = None,
    name_columns: Optional[List[str]] = None,
    report: str = "raw",
    confidence_reference: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Clean, gate and score raw descriptions at variable depth.

    thresholds is {depth: raw prefix probability}, e.g.
    {4: 0.90, 3: 0.75, 2: 0.55, 1: 0.30}. A level may be omitted to take
    it out of service entirely.

    cleaning_cfg is the recipe used on the training text, and getting it
    wrong is the most consequential mistake available here. Two forms are
    accepted:

      * a DescriptionCleaner -- anything exposing .clean_series(). This is
        what cleaning.build_cleaner() returns and what save_bundle stores,
        and it carries its own word list so scoring cannot pick up a
        different one than training used.
      * a CleaningConfig, for models trained with the simpler recipe.

    Passing None scores the text as given, which is only correct if the
    model was trained on uncleaned text. It is never the safe default:
    the vocabulary was fitted on cleaned descriptions, so raw text looks
    almost entirely out-of-vocabulary and the failure shows up as a high
    OOV rate rather than an error.

    min_known_tokens is the readability gate. The default of 1 refuses
    only descriptions with no recognisable word at all. Raising it to 2 is
    defensible when descriptions are short, but check how many rows it
    turns away first.

    report changes only what is *displayed*. Every gate and every depth
    decision is made on the raw score regardless, and that raw score stays
    in the `confidence` column. See _reported_confidence for the modes;
    "percentile" is the one to prefer when the figure sits next to a
    decision, and it needs confidence_reference.

    Returns one row per input:

        raw_text, cleaned_text     what arrived, and what was scored
        n_tokens, n_known          words found, and words the model knows
        oov_rate                   share unrecognised
        status                     'predicted' or one of the refusals
        assigned_depth             0 when refused
        confidence                 raw score at the assigned depth -- the
                                   value the decision was actually made on,
                                   always present so the decision stays
                                   auditable
        reported_confidence        the same thing expressed per `report`
        confidence_basis           which mode produced it
        <level columns>            filled to the assigned depth, blank below
        <name columns>            optional, when lookup is supplied
    """
    from .cleaning.text import clean_series
    from .quality import describe_inputs

    levels = level_columns or clf.level_columns
    raw = pd.Series(list(raw_texts), dtype="object").fillna("").astype(str)

    if cleaning_cfg is None:
        cleaned = raw.copy()
    elif hasattr(cleaning_cfg, "clean_series"):
        # DescriptionCleaner: the full procedure, word list included.
        cleaned = cleaning_cfg.clean_series(raw)
    else:
        cleaned = clean_series(raw, cleaning_cfg)

    quality = describe_inputs(clf, cleaned)
    readable = (quality["n_known"] >= min_known_tokens).to_numpy()

    # Scored on the cleaned text, and explicitly on the raw scale: the
    # thresholds are raw, and predict_prefix_probabilities normalises by
    # default, which would silently compare two different quantities.
    prefix = clf.predict_prefix_probabilities(cleaned, normalise=False)

    n = len(raw)
    depth = np.zeros(n, dtype=int)
    conf = np.full(n, np.nan)
    for d in sorted((d for d in thresholds if d <= len(levels)), reverse=True):
        s = prefix[PREFIX_PROB_FMT.format(d=d)].to_numpy(dtype=float)
        take = (depth == 0) & readable & (s >= thresholds[d])
        depth[take] = d
        conf[take] = s[take]

    status = np.where(
        ~readable, REFUSED_UNREADABLE,
        np.where(depth == 0, REFUSED_UNCERTAIN, "predicted"),
    )

    out = {c: np.full(n, "", dtype=object) for c in levels}
    for d in sorted(set(depth[depth > 0])):
        rows = depth == d
        for c in levels[:d]:
            out[c][rows] = prefix.loc[rows, PRED_FMT.format(d=d, level=c)].astype(str)

    frame = pd.DataFrame({
        "raw_text": raw.to_numpy(),
        "cleaned_text": cleaned.to_numpy(),
        "n_tokens": quality["n_tokens"].to_numpy(),
        "n_known": quality["n_known"].to_numpy(),
        "oov_rate": quality["oov_rate"].to_numpy(),
        "status": status,
        "assigned_depth": depth,
        "confidence": conf,
        **out,
    })
    frame["levels_left_to_do"] = len(levels) - frame["assigned_depth"]

    normalised = None
    if report == "model_normalised":
        # A second pass, on the normalised scale. Only ever used for the
        # displayed figure -- the decision above is already made.
        norm = clf.predict_prefix_probabilities(cleaned, normalise=True)
        normalised = np.full(n, np.nan)
        for d in sorted(set(depth[depth > 0])):
            rows = depth == d
            normalised[rows] = norm.loc[
                rows, PREFIX_PROB_FMT.format(d=d)
            ].to_numpy(dtype=float)

    frame["reported_confidence"] = _reported_confidence(
        conf, depth, thresholds, report,
        reference=confidence_reference, normalised=normalised,
    )
    frame["confidence_basis"] = np.where(depth > 0, report, "")

    if lookup is not None and name_columns:
        from .artifacts import attach_names

        named = attach_names(frame, lookup, levels, name_columns)
        for c in name_columns:
            # Blank rather than NaN where nothing was predicted at all.
            frame[c] = named[c].where(named[levels[0]] != "", "")
    return frame


def predict_all_depths(
    clf: HierarchicalPathClassifier,
    raw_texts,
    cleaning_cfg=None,
    level_columns: Optional[List[str]] = None,
    lookup: Optional[pd.DataFrame] = None,
    name_columns: Optional[List[str]] = None,
    min_known_tokens: int = 1,
    prefix: str = "Predicted ",
) -> pd.DataFrame:
    """Predict the full path, and report raw confidence at every level.

    The input needs a description and nothing else. Category columns are
    what this produces, so requiring them would refuse the unlabelled
    records it exists to handle.

    Predicted columns are named with `prefix`, so they neither collide
    with nor get mistaken for existing label columns when a file already
    carries some -- which is what you want when comparing a stale label
    against a fresh prediction. Pass prefix="" to name them bare.

    For a batch of records rather than a threshold decision: no depth is
    chosen and nothing is refused, so every input gets a prediction and
    the confidence figures needed to decide what to do with it.

    Confidence is the **raw, uncalibrated** prefix probability -- the
    product of the per-level posteriors, summed over paths sharing that
    prefix. It is not comparable between levels: a depth-1 figure and a
    depth-4 figure come from different distributions, so a smaller number
    at one depth is not a lower bar than a larger number at another.
    Compare a level against its own threshold, never against another
    level.

    One subtlety the output makes visible. `L{d} Confidence` belongs to
    the best prefix *at that depth*, which is not always the first d
    levels of the full-depth answer: summing over a prefix's children can
    let a different prefix win. `L{d} Agrees` flags where the two part
    company -- worth looking at, since those are the products where
    truncating would change the answer rather than just shorten it.

    Readability is reported, not enforced. A description with no
    recognisable word still gets a prediction here, drawn from category
    frequencies rather than from the product; `readable` is False on those
    rows and they should not be accepted whatever their confidence says.
    """
    from .quality import describe_inputs

    levels = level_columns or clf.level_columns
    depth = len(levels)

    raw = pd.Series(list(raw_texts), dtype="object").fillna("").astype(str)
    if cleaning_cfg is None:
        cleaned = raw.copy()
    elif hasattr(cleaning_cfg, "clean_series"):
        cleaned = cleaning_cfg.clean_series(raw)
    else:
        from .cleaning.text import clean_series

        cleaned = clean_series(raw, cleaning_cfg)

    quality = describe_inputs(clf, cleaned)
    prefix_df = clf.predict_prefix_probabilities(cleaned, normalise=False)

    out = pd.DataFrame({
        "raw_text": raw.to_numpy(),
        "cleaned_text": cleaned.to_numpy(),
        "n_tokens": quality["n_tokens"].to_numpy(),
        "n_known": quality["n_known"].to_numpy(),
        "oov_rate": quality["oov_rate"].to_numpy(),
        "readable": (quality["n_known"] >= min_known_tokens).to_numpy(),
    })

    # The full-depth answer is the prediction; shallower depths contribute
    # their confidence and whether they agree.
    for lv in levels:
        out[lv] = prefix_df[PRED_FMT.format(d=depth, level=lv)].astype(str).to_numpy()

    for d in range(1, depth + 1):
        out[f"L{d} Confidence"] = prefix_df[PREFIX_PROB_FMT.format(d=d)].to_numpy(float)
        if d < depth:
            agrees = np.ones(len(out), dtype=bool)
            for lv in levels[:d]:
                agrees &= (
                    prefix_df[PRED_FMT.format(d=d, level=lv)].astype(str).to_numpy()
                    == out[lv].to_numpy()
                )
            out[f"L{d} Agrees"] = agrees

    if lookup is not None and name_columns:
        from .artifacts import attach_names

        out = attach_names(out, lookup, levels, name_columns)

    if prefix:
        renames = {lv: f"{prefix}{lv}" for lv in levels}
        if lookup is not None and name_columns:
            renames.update({nm: f"{prefix}{nm}" for nm in name_columns})
        out = out.rename(columns=renames)
    return out


def scoring_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Counts and shares by status and depth -- what a batch produced."""
    rows = []
    n = len(scored)
    for (status, depth), grp in scored.groupby(["status", "assigned_depth"]):
        rows.append({"status": status, "assigned_depth": depth,
                     "n_items": len(grp), "share": len(grp) / n if n else np.nan})
    out = pd.DataFrame(rows).sort_values(
        ["assigned_depth", "status"], ascending=[False, True]
    )
    out.attrs["n_items"] = n
    out.attrs["refused"] = int((scored["status"] != "predicted").sum())
    return out.reset_index(drop=True)
