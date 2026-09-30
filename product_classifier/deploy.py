"""Refit on everything, then score new products.

The train/test split exists to produce an honest accuracy figure and to
tune thresholds without cheating. Once those numbers are in hand the split
has done its job, and continuing to hold back a quarter of the data just
means shipping a model that has seen less than it could have.

So the sequence is: run() to measure and decide, then fit_final() to
build the model you actually deploy -- same settings, same pipeline
steps, all the rows. On the current golden set that adds roughly 149,000
products and about 4.5% of vocabulary the split model never saw.

Two cautions worth keeping in view:

* The accuracy and thresholds you report belong to the *split* model.
  They carry over to the refitted one only on the assumption that more
  training data does not make it worse, which is safe, and that it does
  not shift the calibration much, which is nearly safe -- a refitted
  model is generally slightly more confident. Re-measuring against fresh
  labelled examples after a few months is the honest correction.
* Never evaluate the refitted model on any of these rows. It has seen all
  of them. Its "accuracy" on them is meaningless.

Nothing here is imported by the rest of the package.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import data as data_mod
from .cleaning.text import CleaningConfig, clean_dataframe
from .config import PipelineConfig
from .model import HierarchicalPathClassifier


def prepare_all(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    cleaning_cfg: Optional[CleaningConfig] = None,
) -> pd.DataFrame:
    """The same cleaning and filtering run() applies, minus the split.

    Deliberately reuses data.drop_null_rows and data.drop_rare_paths
    rather than reimplementing them, so the deployed model is built from
    exactly the rows the evaluated one would have been built from.
    """
    levels = cfg.active_levels()
    if cfg.clean_text:
        df = clean_dataframe(df, cfg.data.text_col, cleaning_cfg)
        print("[final] applied text cleaning")

    df = data_mod.drop_null_rows(
        df,
        cfg.data.text_col,
        levels,
        cfg.data.drop_null_text,
        cfg.data.drop_null_labels,
    )
    df = data_mod.drop_rare_paths(df, levels, cfg.split.min_path_count)
    return df


def fit_final(
    cfg: PipelineConfig,
    df: Optional[pd.DataFrame] = None,
    cleaning_cfg: Optional[CleaningConfig] = None,
    spark=None,
) -> HierarchicalPathClassifier:
    """Train on 100% of the usable data, for production scoring.

    Returns the fitted classifier only -- there is deliberately no
    evaluation attached, because there is nothing left to evaluate it on.
    """
    levels = cfg.active_levels()
    if df is None:
        df = data_mod.load_data(cfg.data, spark=spark)
    print(f"[final] loaded {len(df):,} rows")

    full = prepare_all(cfg, df, cleaning_cfg)
    print(f"[final] training on all {len(full):,} usable rows (no holdout)")

    clf = HierarchicalPathClassifier(level_columns=levels, config=cfg.model)
    clf.fit(full[cfg.data.text_col], full[levels])
    return clf


def score_new(
    clf: HierarchicalPathClassifier,
    texts: pd.Series,
    thresholds: dict,
    level_columns: Optional[list] = None,
    quality_guard: bool = True,
    min_known_tokens: int = 1,
) -> pd.DataFrame:
    """Score unlabelled products end to end, at variable depth.

    This is the production path, and the thing to notice is that no labels
    appear anywhere. Labels were needed once, to choose the thresholds;
    they are not needed again to predict. The thresholds passed here are
    those frozen numbers.

    The quality guard is on by default. Confidence says how sure the model
    is; it does not say whether the model had anything to work with. A
    description whose every word is unknown still scores confidently, on
    nothing but category frequencies, so those rows are handed over
    regardless of what the threshold would allow.
    """
    from .cascade import apply_cascade
    from .quality import apply_input_guard, describe_inputs

    levels = level_columns or clf.level_columns
    prefix = clf.predict_prefix_probabilities(texts)
    chosen = apply_cascade(prefix, levels, thresholds)

    if quality_guard:
        quality = describe_inputs(clf, texts)
        chosen = apply_input_guard(
            chosen, quality, levels, min_known_tokens=min_known_tokens
        )
        chosen = pd.concat(
            [chosen, quality[["n_tokens", "n_known", "oov_rate"]]], axis=1
        )
    return chosen
