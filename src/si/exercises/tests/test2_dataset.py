import numpy as np
from src.si.data.dataset import Dataset


def test_dropna_removes_rows_with_nan():
    X = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
        [4.0, np.nan],
        [5.0, 6.0]
    ])
    y = np.array([0.0, 1.0, np.nan, 0.0])

    ds = Dataset(X, y)
    ds.dropna()

    # after dropna there should be no NaNs in X
    assert not np.isnan(ds.X).any()
    # and no NaNs in y if y is numeric
    assert not np.isnan(ds.y).any()
    # only rows without NaNs remain: first and last
    assert ds.X.shape == (2, 2)
    assert ds.y.shape == (2,)


def test_fillna_mean_replaces_nans_in_X():
    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
        [5.0, np.nan]
    ])
    ds = Dataset(X)

    ds.fillna("mean")

    # no NaNs left in X
    assert not np.isnan(ds.X).any()

    # check if second column values are the mean where there were NaNs
    col = np.array([1.0, 4.0, 4.0])  # mean of [4] is 4
    assert np.allclose(ds.X[:, 1], col)


def test_remove_by_index_removes_correct_row():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    y = np.array([0, 1, 2])

    ds = Dataset(X, y)
    ds.remove_by_index(1)

    # row with index 1 should be gone
    assert ds.X.shape == (2, 2)
    assert ds.y.shape == (2,)
    assert np.array_equal(ds.X, np.array([[1.0, 2.0], [5.0, 6.0]]))
    assert np.array_equal(ds.y, np.array([0, 2]))
