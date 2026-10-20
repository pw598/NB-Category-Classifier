"""SparseMultinomialNB / HierarchicalNB must match scikit-learn exactly.

Run on Databricks or locally:  pytest -q tests
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB

from nbclassifier.config import HyperParams
from nbclassifier.model import HierarchicalNB, SparseMultinomialNB

RTOL, ATOL = 1e-9, 1e-9


def _data(seed=0, n=600, F=80, C=7, density=0.08):
    rng = np.random.default_rng(seed)
    X = sp.random(n, F, density=density, random_state=seed, format="csr") * 3
    y = rng.integers(0, C, n)
    y[:C] = np.arange(C)  # every class present
    return X, y


@pytest.mark.parametrize("alpha", [1.0, 0.1, 0.001])
@pytest.mark.parametrize("fit_prior", [True, False])
def test_flat_matches_sklearn(alpha, fit_prior):
    X, y = _data()
    ours = SparseMultinomialNB(alpha, fit_prior).fit(X, y, n_classes=7)
    ref = MultinomialNB(alpha=alpha, fit_prior=fit_prior, force_alpha=True).fit(X, y)
    np.testing.assert_allclose(ours.feature_log_prob_, ref.feature_log_prob_, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ours.class_log_prior_, ref.class_log_prior_, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ours.joint_log_likelihood(X), ref._joint_log_likelihood(X), rtol=RTOL, atol=1e-7)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-7, atol=1e-10)
    assert (ours.predict(X) == ref.predict(X)).all()


def test_with_params_reuses_counts():
    X, y = _data(1)
    base = SparseMultinomialNB(1.0, True).fit(X, y)
    other = base.with_params(0.01, False)
    ref = MultinomialNB(alpha=0.01, fit_prior=False, force_alpha=True).fit(X, y)
    np.testing.assert_allclose(other.predict_proba(X), ref.predict_proba(X), rtol=1e-7, atol=1e-10)
    assert other.feature_count_ is base.feature_count_


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

LEVELS = ["L1", "L2", "L3"]


def _tree_data(seed=0, n=900, F=60):
    rng = np.random.default_rng(seed)
    paths = [("A", "A1", "A1a"), ("A", "A1", "A1b"), ("A", "A2", "A2a"),
             ("B", "B1", "B1a"), ("B", "B1", "B1b"), ("B", "B1", "B1c"), ("C", "C1", "C1a")]
    idx = rng.integers(0, len(paths), n)
    idx[: len(paths)] = np.arange(len(paths))
    labels = pd.DataFrame([paths[i] for i in idx], columns=LEVELS)
    X = sp.random(n, F, density=0.1, random_state=seed, format="csr")
    X = X + sp.csr_matrix((np.ones(n), (np.arange(n), idx * 3 % F)), shape=(n, F))  # signal
    return X.tocsr(), labels


class _Identity:
    """Stands in for a fitted vectorizer when X is supplied directly."""


def _fit(mode, alpha=0.1, fit_prior=True):
    X, labels = _tree_data()
    hp = HyperParams(prediction_mode=mode, alpha=alpha, fit_prior=fit_prior)
    clf = HierarchicalNB(LEVELS, hp).fit(None, labels, vectorizer=_Identity(), X=X)
    return clf, X, labels


def _sk(X, y, alpha, fit_prior):
    return MultinomialNB(alpha=alpha, fit_prior=fit_prior, force_alpha=True).fit(X, y)


def _paths(clf):
    return [tuple(r) for r in clf.valid_paths().itertuples(index=False)]


@pytest.mark.parametrize("fit_prior", [True, False])
def test_independent_matches_sklearn(fit_prior):
    clf, X, labels = _fit("independent", fit_prior=fit_prior)
    joint = clf.joint_log(X)
    models = {c: _sk(X, labels[c], 0.1, fit_prior) for c in LEVELS}
    probas = {c: pd.DataFrame(np.log(models[c].predict_proba(X)), columns=models[c].classes_) for c in LEVELS}
    expect = np.column_stack([sum(probas[c][p[d]].to_numpy() for d, c in enumerate(LEVELS)) for p in _paths(clf)])
    np.testing.assert_allclose(joint, expect, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("fit_prior", [True, False])
def test_conditional_matches_per_parent_sklearn(fit_prior):
    """Chain rule with one sklearn MultinomialNB per parent node (full vocabulary)."""
    clf, X, labels = _fit("conditional", fit_prior=fit_prior)
    joint = clf.joint_log(X)
    n = X.shape[0]
    expect = np.zeros((n, len(_paths(clf))))
    for j, p in enumerate(_paths(clf)):
        total = np.zeros(n)
        for d, col in enumerate(LEVELS):
            mask = np.ones(len(labels), bool)
            for k in range(d):
                mask &= (labels[LEVELS[k]] == p[k]).to_numpy()
            children = labels.loc[mask, col]
            if children.nunique() == 1:
                continue  # single child: probability 1
            m = _sk(X[mask], children, 0.1, fit_prior)
            lp = np.log(m.predict_proba(X))
            total += lp[:, list(m.classes_).index(p[d])]
        expect[:, j] = total
    np.testing.assert_allclose(joint, expect, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(np.exp(joint).sum(axis=1), 1.0, rtol=1e-9)


def test_flattened_matches_sklearn():
    clf, X, labels = _fit("flattened")
    joint = clf.joint_log(X)
    m = _sk(X, labels[LEVELS[-1]], 0.1, True)  # deepest labels are unique per path here
    lp = pd.DataFrame(np.log(m.predict_proba(X)), columns=m.classes_)
    expect = np.column_stack([lp[p[-1]].to_numpy() for p in _paths(clf)])
    np.testing.assert_allclose(joint, expect, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("mode", ["independent", "conditional", "flattened"])
def test_confidence_and_margin(mode):
    clf, X, labels = _fit(mode)
    best, conf, margin = clf.predict_arrays(X, batch_size=97)  # odd batch size on purpose
    scores = np.exp(clf.joint_log(X))
    paths = clf.valid_paths()
    np.testing.assert_array_equal(best, scores.argmax(axis=1))
    for d, col in enumerate(LEVELS):
        prefix = paths[LEVELS[: d + 1]].astype(str).agg("|".join, axis=1).to_numpy()
        for i in range(0, X.shape[0], 53):
            mine = prefix[best[i]]
            mass = pd.Series(scores[i]).groupby(prefix).sum()
            assert conf[i, d] == pytest.approx(mass[mine], rel=1e-9, abs=1e-12)
            rest = mass.drop(mine)
            other = rest.max() if len(rest) else 0.0
            assert margin[i, d] == pytest.approx(mass[mine] - other, rel=1e-9, abs=1e-12)
    # deepest confidence is the predicted path's own score
    np.testing.assert_allclose(conf[:, -1], scores[np.arange(len(best)), best], rtol=1e-12)


def test_clone_with_matches_fresh_fit():
    clf, X, labels = _fit("independent", alpha=1.0)
    clone = clf.clone_with(0.01, False, "conditional")
    fresh, _, _ = _fit("conditional", alpha=0.01, fit_prior=False)
    np.testing.assert_allclose(clone.joint_log(X), fresh.joint_log(X), rtol=1e-10, atol=1e-10)
