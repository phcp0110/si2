"""
Tests for Exercise 3: SelectPercentile
"""

import numpy as np
from src.si.data.dataset import Dataset
from src.si.feature_selection.select_percentile import SelectPercentile


def dummy_score_func(dataset: Dataset):
    """
    Returns fixed F scores for 10 features; p-values constant (unused for selection).
    Matches the tie scenario from the slides:
    [1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2]
    """
    F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
    p = np.ones_like(F) * 0.05
    return F, p


def test_percentile_count_and_ties_match_slide_example():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 10))
    ds = Dataset(X)

    sel = SelectPercentile(score_func=dummy_score_func, percentile=40).fit(ds)
    ds_t = sel.transform(ds)

    # 40% of 10 -> 4 features
    assert ds_t.X.shape[1] == 4
    # ensure same number of rows (samples) is preserved
    assert ds_t.X.shape[0] == ds.X.shape[0]


def test_extreme_percentiles_zero_and_full():
    X = np.random.RandomState(1).randn(8, 6)
    ds = Dataset(X)

    s0 = SelectPercentile(score_func=dummy_score_func, percentile=0).fit(ds)
    ds0 = s0.transform(ds)
    assert ds0.X.shape[1] == 0

    s100 = SelectPercentile(score_func=dummy_score_func, percentile=100).fit(ds)
    ds100 = s100.transform(ds)
    assert ds100.X.shape[1] == X.shape[1]


def test_transform_keeps_label_and_features_names():
    X = np.arange(30.0).reshape(5, 6)
    y = np.array([0, 1, 0, 1, 0])
    feats = [f"f{i}" for i in range(6)]
    ds = Dataset(X, y, feats, label="target")

    sel = SelectPercentile(score_func=dummy_score_func, percentile=50).fit(ds)
    ds_t = sel.transform(ds)

    # y and label preserved
    assert ds_t.y is not None
    assert ds_t.label == "target"
    # same number of samples
    assert ds_t.X.shape[0] == ds.X.shape[0]
    # feature names length matches selected columns
    assert len(ds_t.features) == ds_t.X.shape[1]
