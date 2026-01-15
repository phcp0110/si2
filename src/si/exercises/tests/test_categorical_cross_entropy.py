# tests/test_categorical_cross_entropy.py
import numpy as np
import pytest

from si.neural_networks.losses import CategoricalCrossEntropy


def test_loss_raises_on_shape_mismatch():
    loss_fn = CategoricalCrossEntropy()

    y_true = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=float)
    y_pred = np.array([[0.7, 0.2, 0.1]], dtype=float)  

    with pytest.raises(ValueError):
        _ = loss_fn.loss(y_true, y_pred)

    with pytest.raises(ValueError):
        _ = loss_fn.derivative(y_true, y_pred)


def test_loss_known_simple_case_one_hot():
    """
    If y_true selects class k, CCE per sample is -log(p_k).
    """
    loss_fn = CategoricalCrossEntropy()

    y_true = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=float)
    y_pred = np.array([[0.8, 0.1, 0.1],
                       [0.2, 0.5, 0.3]], dtype=float)

    expected = (-np.log(0.8) - np.log(0.5)) / 2.0
    got = loss_fn.loss(y_true, y_pred)

    assert np.isclose(got, expected, atol=1e-12)


def test_loss_invariant_to_non_true_classes_when_one_hot():
    """
    With one-hot y_true, only the probability of the true class matters.
    """
    loss_fn = CategoricalCrossEntropy()

    y_true = np.array([[0, 0, 1]], dtype=float)

    y_pred1 = np.array([[0.1, 0.2, 0.7]], dtype=float)
    y_pred2 = np.array([[0.25, 0.05, 0.7]], dtype=float)  

    assert np.isclose(loss_fn.loss(y_true, y_pred1), loss_fn.loss(y_true, y_pred2), atol=1e-12)


def test_loss_is_finite_with_zeros_and_ones_due_to_clipping():
    """
    Verifies np.clip avoids log(0) issues.
    """
    loss_fn = CategoricalCrossEntropy()

    y_true = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=float)

    y_pred = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]], dtype=float)

    val = loss_fn.loss(y_true, y_pred)
    assert np.isfinite(val)


def test_derivative_matches_formula_with_clipping():
    loss_fn = CategoricalCrossEntropy()

    y_true = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=float)
    y_pred = np.array([[0.8, 0.1, 0.1],
                       [0.2, 0.5, 0.3]], dtype=float)

    grad = loss_fn.derivative(y_true, y_pred)
    expected = -y_true / y_pred 

    assert grad.shape == y_pred.shape
    assert np.allclose(grad, expected, atol=1e-12)


def test_derivative_is_finite_with_zeros_due_to_clipping():
    """
    Verifies np.clip avoids division by 0.
    """
    loss_fn = CategoricalCrossEntropy()

    y_true = np.array([[1, 0, 0]], dtype=float)
    y_pred = np.array([[0.0, 0.5, 0.5]], dtype=float)  

    grad = loss_fn.derivative(y_true, y_pred)

    assert np.isfinite(grad).all()
    assert grad[0, 0] < 0
    assert grad[0, 1] == 0
    assert grad[0, 2] == 0


def test_loss_decreases_when_true_class_probability_increases():
    """
    Sanity check: increasing p_true should reduce cross-entropy.
    """
    loss_fn = CategoricalCrossEntropy()

    y_true = np.array([[0, 1, 0]], dtype=float)

    y_pred_low = np.array([[0.45, 0.10, 0.45]], dtype=float)
    y_pred_high = np.array([[0.10, 0.80, 0.10]], dtype=float)

    assert loss_fn.loss(y_true, y_pred_high) < loss_fn.loss(y_true, y_pred_low)


def test_loss_returns_python_float():
    loss_fn = CategoricalCrossEntropy()

    y_true = np.array([[1, 0]], dtype=float)
    y_pred = np.array([[0.9, 0.1]], dtype=float)

    val = loss_fn.loss(y_true, y_pred)
    assert isinstance(val, float)
