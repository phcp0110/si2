import numpy as np
import pytest

from si.data.dataset import Dataset

from si.model_selection import randomized_search


class DummyModel:
    """
    Minimal model with hyperparameters as attributes.
    randomized_search_cv uses hasattr/setattr on these.
    """

    def __init__(self):
        self.alpha = 0.0
        self.max_iter = 0


def test_raises_on_invalid_hyperparameter_name():
    model = DummyModel()
    ds = Dataset(X=np.random.randn(6, 2), y=np.array([0, 1, 0, 1, 0, 1]))

    grid = {"does_not_exist": np.array([1, 2, 3])}

    with pytest.raises(ValueError):
        randomized_search(
            model=model,
            dataset=ds,
            hyperparameter_grid=grid,
            scoring=lambda y_true, y_pred: 1.0,
            cv=3,
            n_iter=5,
            random_state=0,
        )


def test_output_structure_and_lengths(monkeypatch):
    """
    We monkeypatch k_fold_cross_validation inside the module under test
    so the test is fast and isolates the randomized_search_cv logic.
    """
    import si.model_selection.randomized_search as rs_mod

    def fake_kfold_cv(model, dataset, scoring, k):
        return [float(model.alpha) + float(model.max_iter) / 1000.0] * k

    monkeypatch.setattr(rs_mod, "k_fold_cross_validation", fake_kfold_cv)

    model = DummyModel()
    ds = Dataset(X=np.random.randn(10, 2), y=np.random.randint(0, 2, size=10))

    grid = {
        "alpha": np.array([0.1, 0.2, 0.3]),
        "max_iter": np.array([500, 1000]),
    }

    res = randomized_search(
        model=model,
        dataset=ds,
        hyperparameter_grid=grid,
        scoring=lambda y_true, y_pred: 1.0,
        cv=3,
        n_iter=7,
        random_state=42,
    )

    assert set(res.keys()) == {
        "hyperparameters",
        "scores",
        "best_hyperparameters",
        "best_score",
    }
    assert len(res["hyperparameters"]) == 7
    assert len(res["scores"]) == 7
    assert res["best_score"] == max(res["scores"])
    assert isinstance(res["best_hyperparameters"], dict)


def test_hyperparameters_sampled_from_grid(monkeypatch):
    import si.model_selection.randomized_search as rs_mod

    def fake_kfold_cv(model, dataset, scoring, k):
        return [0.0] * k

    monkeypatch.setattr(rs_mod, "k_fold_cross_validation", fake_kfold_cv)

    model = DummyModel()
    ds = Dataset(X=np.random.randn(8, 3), y=np.random.randint(0, 2, size=8))

    grid = {"alpha": np.array([1.0, 2.0]), "max_iter": np.array([10, 20, 30])}

    res = randomized_search(
        model=model,
        dataset=ds,
        hyperparameter_grid=grid,
        scoring=lambda y_true, y_pred: 1.0,
        cv=4,
        n_iter=20,
        random_state=1,
    )

    for p in res["hyperparameters"]:
        assert p["alpha"] in grid["alpha"]
        assert p["max_iter"] in grid["max_iter"]


def test_random_state_makes_results_repeatable(monkeypatch):
    import si.model_selection.randomized_search as rs_mod

    def fake_kfold_cv(model, dataset, scoring, k):
        return [float(model.alpha) * 10 + float(model.max_iter)] * k

    monkeypatch.setattr(rs_mod, "k_fold_cross_validation", fake_kfold_cv)

    model1, model2 = DummyModel(), DummyModel()
    ds = Dataset(X=np.random.randn(12, 2), y=np.random.randint(0, 2, size=12))

    grid = {"alpha": np.array([0.1, 0.2, 0.3]), "max_iter": np.array([1, 2, 3])}

    res1 = randomized_search(
        model=model1,
        dataset=ds,
        hyperparameter_grid=grid,
        scoring=lambda y_true, y_pred: 1.0,
        cv=3,
        n_iter=10,
        random_state=123,
    )
    res2 = randomized_search(
        model=model2,
        dataset=ds,
        hyperparameter_grid=grid,
        scoring=lambda y_true, y_pred: 1.0,
        cv=3,
        n_iter=10,
        random_state=123,
    )

    assert res1["hyperparameters"] == res2["hyperparameters"]
    assert res1["scores"] == res2["scores"]
    assert res1["best_hyperparameters"] == res2["best_hyperparameters"]
    assert res1["best_score"] == res2["best_score"]
