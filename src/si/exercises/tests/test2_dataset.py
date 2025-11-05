"""
Tests for Exercise 2: Dataset methods
- dropna()
- fillna()
- remove_by_index()
"""

import numpy as np
import pytest
from src.si.data.dataset import Dataset


def test_dropna_removes_rows_with_nan_in_X_and_numeric_y():
    X = np.array([[1.0, 2.0], [np.nan, 5.0], [3.0, np.nan], [4.0, 4.0]])
    y = np.array([0.0, 1.0, np.nan, 0.0])
    ds = Dataset(X, y).dropna()
    assert ds.X.shape == (2, 2)
    assert ds.y.shape == (2,)
    assert not np.isnan(ds.X).any()
    assert not np.isnan(ds.y).any()


def test_fillna_with_scalar_and_statistics():
    X = np.array([[1.0, np.nan], [np.nan, 3.0]])

    # scalar
    ds_scalar = Dataset(X.copy()).fillna(0.5)
    assert np.allclose(ds_scalar.X, np.array([[1.0, 0.5], [0.5, 3.0]]))

    # mean
    ds_mean = Dataset(X.copy()).fillna("mean")
    col_means = np.nanmean(X, axis=0)
    expected_mean = X.copy()
    expected_mean[np.isnan(expected_mean[:, 0]), 0] = col_means[0]
    expected_mean[np.isnan(expected_mean[:, 1]), 1] = col_means[1]
    assert np.allclose(ds_mean.X, expected_mean, equal_nan=False)

    # median
    ds_median = Dataset(X.copy()).fillna("median")
    col_median = np.nanmedian(X, axis=0)
    expected_med = X.copy()
    expected_med[np.isnan(expected_med[:, 0]), 0] = col_median[0]
    expected_med[np.isnan(expected_med[:, 1]), 1] = col_median[1]
    assert np.allclose(ds_median.X, expected_med, equal_nan=False)


def test_remove_by_index_and_bounds():
    X = np.arange(12.0).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    ds = Dataset(X, y).remove_by_index(2)
    assert ds.X.shape == (5, 2)
    assert ds.y.shape == (5,)
    # the removed row [4., 5.] should not exist anymore
    assert not np.any(np.all(ds.X == np.array([4.0, 5.0]), axis=1))

    with pytest.raises(IndexError):
        Dataset(np.zeros((3, 2))).remove_by_index(3)
