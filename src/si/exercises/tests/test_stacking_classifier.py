import numpy as np
import pytest

from si.data.dataset import Dataset

from si.ensemble import stacking_classifier


class DummyClassifier:
    """
    Minimal classifier compatible with your framework usage:
    implements _fit and _predict.
    """

    def __init__(self, constant_pred: int):
        self.constant_pred = constant_pred
        self.fit_calls = 0
        self.last_fit_dataset = None
        self.last_predict_dataset = None

    def _fit(self, dataset: Dataset):
        self.fit_calls += 1
        self.last_fit_dataset = dataset
        return self

    def _predict(self, dataset: Dataset):
        self.last_predict_dataset = dataset
        n = dataset.X.shape[0]
        return np.full(n, self.constant_pred, dtype=int)


class SpyFinalModel(DummyClassifier):
    """Final model that records the X it was trained/predicted on."""

    def __init__(self):
        super().__init__(constant_pred=0)

    def _fit(self, dataset: Dataset):
        super()._fit(dataset)
        y = np.asarray(dataset.y).reshape(-1)
        self.constant_pred = int(np.round(y.mean())) if y.size else 0
        return self


def test_init_validations():
    m1 = DummyClassifier(0)
    fm = SpyFinalModel()

    with pytest.raises(ValueError):
        stacking_classifier(models=[], final_model=fm)

    with pytest.raises(ValueError):
        stacking_classifier(models=[m1], final_model=None)


def test_fit_requires_labels():
    m1 = DummyClassifier(0)
    fm = SpyFinalModel()
    clf = stacking_classifier([m1], fm)

    ds = Dataset(X=np.ones((5, 2)), y=None)
    with pytest.raises(ValueError):
        clf._fit(ds)


def test_meta_features_shape_and_content():
    m1 = DummyClassifier(0)
    m2 = DummyClassifier(1)
    fm = SpyFinalModel()
    clf = stacking_classifier([m1, m2], fm)

    ds = Dataset(X=np.random.randn(4, 3), y=np.array([0, 1, 0, 1]))
    Z = clf._meta_features(ds)

    assert Z.shape == (4, 2)
    assert np.all(Z[:, 0] == 0)
    assert np.all(Z[:, 1] == 1)


def test_fit_trains_base_models_and_final_model_on_meta_dataset():
    m1 = DummyClassifier(0)
    m2 = DummyClassifier(1)
    fm = SpyFinalModel()
    clf = stacking_classifier([m1, m2], fm)

    X = np.random.randn(6, 2)
    y = np.array([0, 1, 1, 0, 1, 1])
    ds = Dataset(X=X, y=y)

    clf._fit(ds)

    assert m1.fit_calls == 1
    assert m2.fit_calls == 1
    assert m1.last_fit_dataset is ds
    assert m2.last_fit_dataset is ds

    assert fm.fit_calls == 1
    assert fm.last_fit_dataset.X.shape == (6, 2)  
    assert np.array_equal(fm.last_fit_dataset.y, y)


def test_predict_uses_meta_features_and_final_model():
    m1 = DummyClassifier(0)
    m2 = DummyClassifier(1)
    fm = SpyFinalModel()
    clf = stacking_classifier([m1, m2], fm)

    X = np.random.randn(5, 2)
    y = np.array([0, 1, 0, 1, 1])
    train = Dataset(X=X, y=y)
    clf._fit(train)

    X2 = np.random.randn(3, 2)
    test = Dataset(X=X2, y=None)
    y_pred = clf._predict(test)

    assert y_pred.shape == (3,)
    assert fm.last_predict_dataset.X.shape == (3, 2)


def test_score_returns_accuracy_float():
    m1 = DummyClassifier(0)
    fm = DummyClassifier(1)
    clf = stacking_classifier([m1], fm)

    X = np.random.randn(4, 2)
    y = np.array([1, 1, 0, 1])
    ds = Dataset(X=X, y=y)

    clf._fit(ds)
    score = clf._score(ds)

    assert isinstance(score, float)
    assert score == 0.75
