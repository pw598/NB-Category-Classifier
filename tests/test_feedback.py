"""Feedback-loop experiments (nbclassifier/feedback.py).

The key property: no organic test-fold row ever reaches a teacher or a student.
"""

import numpy as np
import pandas as pd
import pytest

from nbclassifier import feedback as fb
from nbclassifier import paths
from nbclassifier.config import HyperParams

LEVELS = ["L1", "L2", "L3"]
TREE = [("A", "A1", "A1a"), ("A", "A1", "A1b"), ("A", "A2", "A2a"), ("B", "B1", "B1a"),
        ("B", "B1", "B1b"), ("B", "B2", "B2a"), ("C", "C1", "C1a"), ("C", "C1", "C1b")]


def _rows(n, start, rng, noise):
    idx = rng.integers(0, len(TREE), n)
    out = []
    for i, j in enumerate(idx):
        p = TREE[j] if rng.random() > noise else TREE[rng.integers(len(TREE))]
        words = [f"w{lv.lower()}{rng.integers(2)}" for lv in p] + [f"g{rng.integers(30)}"]
        out.append({"ID": str(start + i), "Clean": " ".join(words) + f" uniq{start + i}",
                    **dict(zip(LEVELS, TREE[j]))})
    return pd.DataFrame(out)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    rng = np.random.default_rng(0)
    organic = _rows(900, 1000, rng, 0.2)
    organic[fb.ROW] = np.arange(len(organic))
    pool = _rows(500, 9000, rng, 0.3)[["ID", "Clean"]]
    pool["derived"] = True
    fd = fb.FeedbackData(organic, pool.copy(), pool, {})
    hp = HyperParams(vectorizer_kind="count", min_df=1, max_df=1.0, stop_words=None, ngram_range=(1, 1),
                     alpha=0.1, rare_path_threshold=1, prediction_mode="conditional")
    cols = fb.Cols("ID", "Clean", LEVELS)
    rule = fb.SelectionRule(level=3, min_confidence=None)
    cfg = {"hyperparameters": hp.to_dict(), "n_folds": 3, "fold_seed": 0, "sample_seed": 0,
           "rule": rule.to_dict(), "columns": cols.__dict__}
    return fd, hp, cols, rule, cfg


def _spy(monkeypatch):
    seen = []
    real = fb.fit_model

    def spy(train, hp, cols):
        seen.append(set(train[cols.cleaned]))
        return real(train, hp, cols)

    monkeypatch.setattr(fb, "fit_model", spy)
    return seen


def test_hash_uniform_nested_and_stable():
    ids = [str(i) for i in range(1000)]
    u1 = fb.hash_uniform(ids, "s")
    u2 = fb.hash_uniform(list(reversed(ids)), "s")[::-1]
    np.testing.assert_array_equal(u1, u2)          # depends on the ID only
    assert ((u1 < 0.2) <= (u1 < 0.5)).all()        # nested
    assert abs((u1 < 0.5).mean() - 0.5) < 0.06


def test_ratio_grid_never_trains_on_test_fold(setup, monkeypatch):
    fd, hp, cols, rule, cfg = setup
    seen = _spy(monkeypatch)
    st = fb.Store.create("t", cfg, fd)
    meta = fb.run_ratio_grid(st, fd, hp, cols, n_folds=3, fold_seed=0, folds_to_run=None,
                             organic_fracs=[0.5, 1.0], ratios=[0.25, 1.0], rule=rule)
    folds = fb.fold_assignment(fd.organic, "ID", 3, 0)
    # every fit happened inside one fold's loop; none may contain any row of that fold
    per_fold = len(seen) // 3
    for k in range(3):
        test_texts = set(fd.organic.loc[folds == k, "Clean"])
        for s in seen[k * per_fold:(k + 1) * per_fold]:
            assert not (s & test_texts)
    r = meta[meta["ratio_target"] == 0.25]
    assert np.allclose(r["ratio"], 0.25, atol=0.01)
    assert len(meta) == 3 * 2 * 3


def test_ratio_capped_when_pool_too_small(setup):
    fd, hp, cols, rule, cfg = setup
    st = fb.Store.create("t", cfg, fd)
    meta = fb.run_ratio_grid(st, fd, hp, cols, n_folds=3, fold_seed=0, folds_to_run=[0],
                             organic_fracs=[1.0], ratios=[5.0], rule=rule)
    row = meta[meta["ratio_target"] == 5.0].iloc[0]
    assert row["capped"] and row["n_derived"] == len(fd.pool)


def test_resume_skips_finished_cells(setup, monkeypatch):
    fd, hp, cols, rule, cfg = setup
    st = fb.Store.create("t", cfg, fd)
    fb.run_ratio_grid(st, fd, hp, cols, n_folds=3, fold_seed=0, folds_to_run=[0],
                      organic_fracs=[1.0], ratios=[0.5], rule=rule)
    seen = _spy(monkeypatch)
    st2 = fb.Store.open(st.folder.name, "t", {k: v for k, v in cfg.items() if k != "columns"})
    fb.run_ratio_grid(st2, st2.data(), hp, cols, n_folds=3, fold_seed=0, folds_to_run=[0],
                      organic_fracs=[1.0], ratios=[0.5], rule=rule)
    assert seen == []
    with pytest.raises(ValueError):
        fb.Store.open(st.folder.name, "t", {**cfg, "fold_seed": 1})


def test_noise_dose_dials_noise_and_keeps_test_fold_out(setup, monkeypatch):
    fd, hp, cols, rule, cfg = setup
    seen = _spy(monkeypatch)
    st = fb.Store.create("d", cfg, fd)
    meta = fb.run_noise_dose(st, fd, hp, cols, n_folds=3, fold_seed=0, folds_to_run=[1], ratio=0.5,
                             accept_rate=1.0, doses=[0.0, 0.5, 1.0], rule=rule)
    folds = fb.fold_assignment(fd.organic, "ID", 3, 0)
    test_texts = set(fd.organic.loc[folds == 1, "Clean"])
    assert all(not (s & test_texts) for s in seen)
    d = meta[meta["key"].str.contains("rho")].set_index("dose")
    eta = d["teacher_noise"].iloc[0]
    assert eta > 0
    assert d.loc[0.0, "noise"] == 0
    assert abs(d.loc[1.0, "noise"] - eta) < 1e-12
    assert abs(d.loc[0.5, "noise"] - eta / 2) < 0.01
    assert np.allclose(d["ratio"], 0.5, atol=0.01)


def test_bootstrap_is_paired(setup):
    fd, hp, cols, rule, cfg = setup
    st = fb.Store.create("t", cfg, fd)
    fb.run_ratio_grid(st, fd, hp, cols, n_folds=3, fold_seed=0, folds_to_run=None,
                      organic_fracs=[1.0], ratios=[0.5], rule=rule)
    same = {k: fb.ratio_key(k, 1.0, 0.0) for k in range(3)}
    point, boot = fb.bootstrap_cells(st, [same, same], depth=3, n_boot=50)
    assert np.allclose(boot[:, 0], boot[:, 1]) and point[0] == point[1]
    other = {k: fb.ratio_key(k, 1.0, 0.5) for k in range(3)}
    point, boot = fb.bootstrap_cells(st, [same, other], depth=3, n_boot=200)
    assert boot[:, 0].std() > 0


def test_ratio_surface_recovers_known_curve():
    r = np.array([0.1, 0.5, 1, 2, 0.1, 0.5, 1, 2])
    n = np.array([1000] * 4 + [4000] * 4)
    s = fb.RatioSurface(4000)
    y = s.design(r, n) @ np.array([0.02, -0.01, 0.005])
    s.fit(r, n, y)
    np.testing.assert_allclose(s.predict(0.7, 8000).ravel(), s.design([0.7], [8000]) @ [0.02, -0.01, 0.005])
