"""
Tests for Exercise 1 using the iris.csv dataset.

These tests perform the same operations requested on the slides:
- Penultimate feature extraction
- Mean of the last 10 samples
- Count of samples where ALL features <= 6
- Count of samples not labeled 'Iris-setosa'

If datasets/iris.csv is missing, tests are skipped.
"""

import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("datasets/iris.csv"),
    reason="datasets/iris.csv not found; skipping Exercise 1 tests."
)

from src.si.io.csv_file import read_csv  # noqa: E402


def test_penultimate_feature_shape():
    ds = read_csv("datasets/iris.csv", sep=",", features=True, label=True)
    penultimate = ds.X[:, -2]
    assert penultimate.shape == (ds.X.shape[0],)
    assert penultimate.ndim == 1


def test_last_10_mean_shape_and_values_exist():
    ds = read_csv("datasets/iris.csv", sep=",", features=True, label=True)
    last10 = ds.X[-10:, :]
    means = last10.mean(axis=0)
    assert means.shape == (ds.X.shape[1],)
    assert np.isfinite(means).all()


def test_all_features_leq_6_count_is_int_and_in_range():
    ds = read_csv("datasets/iris.csv", sep=",", features=True, label=True)
    mask = np.all(ds.X <= 6.0, axis=1)
    count = int(mask.sum())
    assert isinstance(count, int)
    # must be between 0 and number of samples
    assert 0 <= count <= ds.X.shape[0]


def test_not_setosa_count_expected():
    ds = read_csv("datasets/iris.csv", sep=",", features=True, label=True)
    # Classic iris has 50 samples per class → not 'Iris-setosa' = 100
    mask_not_setosa = ds.y != "Iris-setosa"
    assert int(mask_not_setosa.sum()) == 100
