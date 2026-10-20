import hashlib

import numpy as np

from nbclassifier.folds import id_hash_folds, stratified_folds


def test_id_hash_matches_reference_formula():
    ids = ["I8IGTHF0H3HT", "S-7657976", "CB7451", " 00892502046 "]
    got = id_hash_folds(ids, n_folds=5, seed=0)
    for key, f in zip(ids, got):
        d = hashlib.sha1(f"0|{key.strip()}".encode("utf-8")).digest()
        assert f == int.from_bytes(d[:8], "big") % 5


def test_stratified_covers_all_folds():
    labels = np.repeat(list("abcde"), 20)
    folds = stratified_folds(labels, 5, seed=1)
    for lab in "abcde":
        assert sorted(np.bincount(folds[labels == lab], minlength=5)) == [4] * 5


def test_id_hash_sample_is_stable_and_about_right():
    from nbclassifier.folds import id_hash_sample

    ids = [f"I{i:06d}" for i in range(20000)]
    m = id_hash_sample(ids, 0.2, seed=0)
    assert 0.18 < m.mean() < 0.22
    # depends on the ID only, not on which other rows are present
    sub = ids[::3]
    assert (id_hash_sample(sub, 0.2, seed=0) == m[::3]).all()
