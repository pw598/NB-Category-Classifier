"""Saving and reloading a trained classifier as a set of files.

What actually has to be kept, and what does not:

    keep   the fitted vectorizer -- its vocabulary and IDF weights are what
           map a new description into the same feature space the models
           were trained in. Without it the models are meaningless.
    keep   one MultinomialNB per level.
    keep   the path index: which hierarchy paths were observed in training,
           and how each level's labels map to model columns. joint_path
           scoring cannot run without it.
    keep   the config, so a later run can be reproduced.
    keep   the cleaning recipe. The vocabulary was built from cleaned text,
           so scoring has to clean identically -- a model given raw text it
           was trained to see cleaned will report most words as unknown and
           quietly fall back to category frequencies.
    drop   the document-term matrix. It is derived data for the training
           rows; regenerating it from text costs seconds, and a stored copy
           can only fall out of step with the vectorizer that made it.

Two layouts are written, deliberately:

    classifier.pkl        the whole fitted object in one file. Simplest to
                          reload, and the vectorizer and models cannot
                          drift apart because they never separate.
    vectorizer.pkl,       the same thing in pieces, for loading one level
    model_L*.pkl,         at a time or inspecting a single component.
    path_index.pkl

Both come from one fit, so they agree on arrival. The risk is later: if
someone retrains and copies only some of the files, the pieces disagree
and the failure is silent -- predictions come out plausible and wrong. The
manifest exists to catch that, and load_bundle checks it.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ModelConfig
from .model import HierarchicalPathClassifier

MANIFEST = "manifest.json"


def _fingerprint(clf: HierarchicalPathClassifier) -> str:
    """A short hash of the shapes that must agree across files.

    Not a security measure -- it is there so that a vectorizer from one
    training run and models from another are rejected loudly instead of
    producing confident nonsense.
    """
    parts = [
        f"features={len(clf.vectorizer.vocabulary_)}",
        f"levels={','.join(clf.level_columns)}",
        f"paths={len(clf.valid_paths_)}",
    ]
    parts += [
        f"{lv}:{len(clf.models[lv].classes_)}"
        for lv in clf.level_columns
        if lv in clf.models
    ]
    stack = getattr(clf, "node_stack_", None)
    if stack:
        parts += [
            f"node{d}:{0 if s['W'] is None else s['W'].shape[1]}"
            for d, s in sorted(stack.items())
        ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _level_tag(level: str) -> str:
    """A filesystem-safe stem from a column name like 'Level 3 ID'."""
    return "".join(ch if ch.isalnum() else "_" for ch in level).strip("_")


def estimate_bundle_size(
    clf: HierarchicalPathClassifier,
    limit_mb: Optional[float] = None,
    include_combined: bool = True,
    drop_fit_state: bool = False,
    float32: bool = False,
) -> pd.DataFrame:
    """Projected size of each file, before writing anything.

    Sizes come from the array shapes, so this is exact for the
    uncompressed case and needs no disk. Compression is not modelled --
    joblib compress=3 typically gets 2-4x on these arrays, but the ratio
    depends on the values.

    What dominates: MultinomialNB stores two dense (n_classes x n_features)
    float64 arrays per level, feature_count_ and feature_log_prob_. Only
    feature_log_prob_ is needed to predict; feature_count_ exists so
    partial_fit can continue, which a deployed model never does.

        drop_fit_state   discards feature_count_       -> about half
        float32          halves what remains
        include_combined False skips classifier.pkl    -> about half again

    Pass limit_mb to get an over_by_mb column, negative meaning it fits.
    """
    itemsize = 4 if float32 else 8
    arrays_per_model = 1 if drop_fit_state else 2

    rows = []
    per_level = {}
    for lv in clf.level_columns:
        n_classes = len(clf.models[lv].classes_)
        n_features = len(clf.vectorizer.vocabulary_)
        size = arrays_per_model * n_classes * n_features * itemsize
        per_level[lv] = size
        rows.append({"file": f"model_{_level_tag(lv)}.pkl",
                     "n_classes": n_classes, "bytes": size})

    vocab = len(clf.vectorizer.vocabulary_)
    # vocabulary dict plus the idf vector; the dict dominates and pickles
    # at roughly 100 bytes per entry once keys and hashing are counted
    rows.append({"file": "vectorizer.pkl", "n_classes": np.nan,
                 "bytes": vocab * 100 + vocab * 8})

    n_paths = len(clf.valid_paths_)
    n_levels = len(clf.level_columns)
    rows.append({"file": "path_index.pkl", "n_classes": np.nan,
                 "bytes": n_paths * n_levels * 60 + n_paths * n_levels * 8})

    if include_combined:
        rows.append({"file": "classifier.pkl", "n_classes": np.nan,
                     "bytes": sum(per_level.values())
                     + vocab * 108 + n_paths * n_levels * 68})

    out = pd.DataFrame(rows)
    out["mb"] = out["bytes"] / 1e6
    if limit_mb is not None:
        out["over_by_mb"] = out["mb"] - limit_mb
        out["fits"] = out["over_by_mb"] <= 0
    total = out["mb"].sum()
    out.attrs["total_mb"] = total
    out.attrs["largest_file_mb"] = out["mb"].max()
    if limit_mb is not None:
        out.attrs["largest_over_by_mb"] = out["mb"].max() - limit_mb
        out.attrs["ratio_of_limit"] = out["mb"].max() / limit_mb
    return out.sort_values("mb", ascending=False).reset_index(drop=True)


def save_bundle(
    clf: HierarchicalPathClassifier,
    directory: str,
    n_training_rows: Optional[int] = None,
    hierarchy_lookup: Optional[pd.DataFrame] = None,
    cleaning_cfg=None,
    confidence_reference: Optional[pd.DataFrame] = None,
    notes: str = "",
    compress: int = 3,
    include_combined: bool = True,
    drop_fit_state: bool = False,
    float32: bool = False,
) -> Dict[str, str]:
    """Write every artifact needed to score new products later.

    hierarchy_lookup is optional but strongly worth passing when the model
    predicts ID columns: the predictions come back as IDs, and a lookup
    saved beside the model is what turns them back into names months later
    without re-querying the source.

    cleaning_cfg should be whatever was used to clean the training text.
    Scoring has to apply the same recipe: the vocabulary was fitted on
    cleaned descriptions, so raw text arriving at a model trained on
    cleaned text looks mostly out-of-vocabulary. Saving it here removes
    the chance of the two drifting apart.

    Size controls, in the order worth trying -- run estimate_bundle_size()
    first to see what each is worth:

        compress          joblib compression level, 0-9. 3 is a good
                          trade; these are dense float arrays and
                          typically shrink 2-4x. No effect on values.
        include_combined  False skips classifier.pkl. That file holds the
                          same data as the pieces, so writing both stores
                          everything twice. Keeping it is only for
                          convenience of reload.
        drop_fit_state    True discards each model's feature_count_. Only
                          feature_log_prob_ is needed to predict; the
                          counts exist so partial_fit can continue, which
                          a deployed model never does. Roughly halves the
                          model files and does not change a prediction.
                          Irreversible for the saved copy -- the model in
                          memory is left alone.
        float32           True halves the log-probability arrays. Unlike
                          the others this *can* shift a prediction, in the
                          rare case where two paths score within float32
                          resolution of each other. Use last.

    Returns {name: path} for everything written.
    """
    import copy

    import joblib

    if not getattr(clf, "fitted_", False):
        raise RuntimeError("Refusing to save an unfitted classifier.")
    os.makedirs(directory, exist_ok=True)
    written = {}

    def put(name, filename, obj):
        path = os.path.join(directory, filename)
        joblib.dump(obj, path, compress=compress)
        written[name] = path

    def slim(model):
        """A copy of one level's model carrying only what prediction needs."""
        if not (drop_fit_state or float32):
            return model
        out = copy.copy(model)
        if float32 and hasattr(out, "feature_log_prob_"):
            out.feature_log_prob_ = out.feature_log_prob_.astype(np.float32)
        if drop_fit_state and hasattr(out, "feature_count_"):
            # class_count_ is kept: it is small, and class_log_prior_ is
            # derived from it.
            out.feature_count_ = np.zeros((0, 0), dtype=np.float32)
        return out

    if include_combined:
        if drop_fit_state or float32:
            trimmed = copy.copy(clf)
            trimmed.models = {lv: slim(m) for lv, m in clf.models.items()}
            put("classifier", "classifier.pkl", trimmed)
        else:
            put("classifier", "classifier.pkl", clf)
    put("vectorizer", "vectorizer.pkl", clf.vectorizer)
    put("config", "config.pkl", clf.config)
    put("path_index", "path_index.pkl", {
        "level_columns": clf.level_columns,
        "valid_paths_": clf.valid_paths_,
        "label_to_col": clf.label_to_col,
        "_path_idx_arrays": clf._path_idx_arrays,
    })
    for lv in clf.level_columns:
        if lv in clf.models:
            put(f"model::{lv}", f"model_{_level_tag(lv)}.pkl", slim(clf.models[lv]))

    # conditional_path only. Held separately so the piecewise reload can
    # rebuild a classifier that still scores, rather than one that looks
    # fine and then cannot find its per-parent factors.
    if getattr(clf, "node_stack_", None):
        put("node_stack", "node_stack.pkl", clf.node_stack_)

    if cleaning_cfg is not None:
        put("cleaning_config", "cleaning_config.pkl", cleaning_cfg)

    if confidence_reference is not None:
        path = os.path.join(directory, "confidence_reference.csv")
        confidence_reference.to_csv(path, index=False)
        written["confidence_reference"] = path

    if hierarchy_lookup is not None:
        path = os.path.join(directory, "hierarchy_lookup.csv")
        hierarchy_lookup.drop_duplicates().to_csv(path, index=False)
        written["hierarchy_lookup"] = path

    import sklearn

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": _fingerprint(clf),
        "level_columns": clf.level_columns,
        "n_features": len(clf.vectorizer.vocabulary_),
        "n_paths": int(len(clf.valid_paths_)),
        "classes_per_level": {lv: int(len(clf.models[lv].classes_))
                              for lv in clf.level_columns if lv in clf.models},
        "n_training_rows": n_training_rows,
        "prediction_mode": clf.config.prediction_mode,
        "vectorizer_kind": clf.config.vectorizer.kind,
        "nb_alpha": clf.config.nb.alpha,
        "max_features": clf.config.vectorizer.max_features,
        "versions": {"scikit-learn": sklearn.__version__,
                     "numpy": np.__version__,
                     "pandas": pd.__version__},
        "files": {k: os.path.basename(v) for k, v in written.items()},
        "has_cleaning_config": cleaning_cfg is not None,
        "has_confidence_reference": confidence_reference is not None,
        "compress": compress,
        "include_combined": include_combined,
        "drop_fit_state": drop_fit_state,
        "float32": float32,
        "notes": notes,
    }
    path = os.path.join(directory, MANIFEST)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    written["manifest"] = path

    # Human-readable copy of the config; the .pkl is what actually reloads.
    from dataclasses import asdict

    path = os.path.join(directory, "config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(clf.config), fh, indent=2, default=str)
    written["config_json"] = path
    return written


def load_bundle(directory: str, from_pieces: bool = False, verify: bool = True):
    """Reload a saved classifier.

    By default this reads classifier.pkl, which cannot be internally
    inconsistent. from_pieces=True rebuilds from the separate files
    instead -- useful if only some were copied, or to confirm the pieces
    agree with the whole.

    verify compares the reloaded object against the manifest fingerprint.
    A mismatch means the files came from different training runs, which is
    worth an exception rather than a warning: mixed artifacts predict
    confidently and wrongly.
    """
    import joblib

    manifest = {}
    manifest_path = os.path.join(directory, MANIFEST)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

    combined = os.path.join(directory, "classifier.pkl")
    if not from_pieces and not os.path.exists(combined):
        # Saved with include_combined=False; the pieces are all there is.
        from_pieces = True

    if not from_pieces:
        clf = joblib.load(combined)
    else:
        config: ModelConfig = joblib.load(os.path.join(directory, "config.pkl"))
        index = joblib.load(os.path.join(directory, "path_index.pkl"))
        clf = HierarchicalPathClassifier(
            level_columns=index["level_columns"], config=config
        )
        clf.vectorizer = joblib.load(os.path.join(directory, "vectorizer.pkl"))
        clf.models = {}
        for lv in index["level_columns"]:
            # Absent when the run was fitted with fit_flat_levels=False.
            piece = os.path.join(directory, f"model_{_level_tag(lv)}.pkl")
            if os.path.exists(piece):
                clf.models[lv] = joblib.load(piece)
        clf.label_to_col = index["label_to_col"]
        clf.valid_paths_ = index["valid_paths_"]
        clf._path_idx_arrays = index["_path_idx_arrays"]
        node_path = os.path.join(directory, "node_stack.pkl")
        if os.path.exists(node_path):
            clf.node_stack_ = joblib.load(node_path)
        clf.fitted_ = True

    if verify and manifest.get("fingerprint"):
        got = _fingerprint(clf)
        if got != manifest["fingerprint"]:
            raise ValueError(
                f"Fingerprint mismatch: manifest says {manifest['fingerprint']}, "
                f"the loaded files give {got}. These artifacts are from "
                "different training runs -- reload a consistent set rather "
                "than scoring with these."
            )
    return clf


def build_confidence_reference(
    prefix_df: pd.DataFrame,
    n_levels: int,
    n_quantiles: int = 1001,
) -> pd.DataFrame:
    """Quantiles of the raw prefix scores, per depth, for reporting.

    Raw scores are the right basis for a decision and a poor thing to show
    a person: they are not comparable between levels, and 0.037 can be a
    high bar or a low one depending on the distribution behind it. Storing
    that distribution lets a raw score be reported as a percentile --
    "higher than 92% of products at this depth" -- which people read
    correctly without being told anything about the scale.

    A percentile is a monotone function of the raw score, so it can never
    disagree with a raw-threshold decision. Normalising the score
    *would*: normalisation divides each row by its own total across
    candidate paths, so it reorders rows, and a refused item could show a
    higher figure than an accepted one.

    Build this from out-of-fold predictions -- notebook 6's `res.prefix`
    is the right population, since those are scores on products the model
    had not seen.
    """
    qs = np.linspace(0.0, 1.0, n_quantiles)
    rows = []
    for d in range(1, n_levels + 1):
        col = f"L{d} Prefix Probability"
        if col not in prefix_df.columns:
            continue
        vals = np.quantile(prefix_df[col].to_numpy(dtype=float), qs)
        rows.extend({"depth": d, "quantile": q, "raw_score": v}
                    for q, v in zip(qs, vals))
    return pd.DataFrame(rows)


def load_confidence_reference(directory: str) -> Optional[pd.DataFrame]:
    """The stored raw-score quantiles, or None if the bundle has none."""
    path = os.path.join(directory, "confidence_reference.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def load_cleaning_config(directory: str):
    """The cleaning recipe saved with the model, or None if absent.

    None means the bundle predates this being saved. Do not simply skip
    cleaning in that case -- reconstruct the recipe the training run used
    and re-save, or every score will be computed on text the model cannot
    read.
    """
    import joblib

    path = os.path.join(directory, "cleaning_config.pkl")
    return joblib.load(path) if os.path.exists(path) else None


def describe_bundle(directory: str) -> pd.DataFrame:
    """What is in a saved bundle, and how big, without loading the models."""
    rows = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            rows.append({"file": name, "size_mb": os.path.getsize(path) / 1e6})
    out = pd.DataFrame(rows)
    manifest_path = os.path.join(directory, MANIFEST)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            out.attrs["manifest"] = json.load(fh)
    return out


def attach_names(
    predictions: pd.DataFrame,
    lookup: pd.DataFrame,
    id_columns: List[str],
    name_columns: List[str],
) -> pd.DataFrame:
    """Add readable names beside predicted ID columns.

    Predicting IDs is the safer choice -- names can repeat under different
    parents, and joint_path matches on string equality -- but IDs are
    unreadable, so this maps them back one level at a time using the
    lookup saved with the model.
    """
    out = predictions.copy()
    for id_col, name_col in zip(id_columns, name_columns):
        pairs = (
            lookup[[id_col, name_col]].astype({id_col: str})
            .drop_duplicates(subset=[id_col])
            .set_index(id_col)[name_col]
        )
        out[name_col] = out[id_col].astype(str).map(pairs)
    return out
