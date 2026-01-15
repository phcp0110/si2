import numpy as np
import pytest

from si.neural_networks.activations import TanhActivation, SoftmaxActivation


def test_tanh_forward_matches_numpy():
    layer = TanhActivation()
    x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]], dtype=float)

    y = layer.forward_propagation(x, training=True)
    assert np.allclose(y, np.tanh(x), atol=1e-12)


def test_tanh_derivative_matches_formula():
    layer = TanhActivation()
    x = np.random.randn(4, 5)

    d = layer.derivative(x)
    expected = 1.0 - np.tanh(x) ** 2
    assert np.allclose(d, expected, atol=1e-12)


def test_tanh_backward_chain_rule_elementwise():
    layer = TanhActivation()
    x = np.random.randn(3, 4)

    _ = layer.forward_propagation(x, training=True)

    output_error = np.ones_like(x)
    grad_in = layer.backward_propagation(output_error)

    expected = 1.0 - np.tanh(x) ** 2
    assert np.allclose(grad_in, expected, atol=1e-12)


def test_tanh_output_shape_and_parameters():
    layer = TanhActivation()

    x = np.random.randn(2, 3)
    _ = layer.forward_propagation(x, training=True)

    assert layer.output.shape == x.shape
    assert layer.parameters() == 0


def test_softmax_forward_rows_sum_to_one():
    layer = SoftmaxActivation()
    x = np.random.randn(6, 10)

    y = layer.forward_propagation(x, training=True)

    row_sums = np.sum(y, axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-12)


def test_softmax_forward_outputs_nonnegative():
    layer = SoftmaxActivation()
    x = np.random.randn(5, 7)

    y = layer.forward_propagation(x, training=True)

    assert np.all(y >= 0.0)


def test_softmax_is_shift_invariant_per_row():
    """
    Softmax(x) == Softmax(x + c) for constant c added to each row.
    """
    layer = SoftmaxActivation()
    x = np.random.randn(4, 8)

    y1 = layer.forward_propagation(x, training=True)
    y2 = layer.forward_propagation(x + 1000.0, training=True)

    assert np.allclose(y1, y2, atol=1e-12)


def test_softmax_numerical_stability_large_values_no_nan_inf():
    layer = SoftmaxActivation()
    x = np.array([[1000.0, 1001.0, 999.0],
                  [1e6, 1e6 + 1.0, 1e6 - 1.0]], dtype=float)

    y = layer.forward_propagation(x, training=True)

    assert np.isfinite(y).all()
    assert np.allclose(np.sum(y, axis=1), np.ones(y.shape[0]), atol=1e-12)


def test_softmax_forward_known_case():
    """
    Simple check: if two logits are equal, probs should be equal.
    """
    layer = SoftmaxActivation()
    x = np.array([[0.0, 0.0]], dtype=float)

    y = layer.forward_propagation(x, training=True)
    assert np.allclose(y, np.array([[0.5, 0.5]]), atol=1e-12)


def test_softmax_derivative_elementwise_bounds():
    """
    With the simplified derivative s*(1-s), values should be in [0, 0.25].
    """
    layer = SoftmaxActivation()
    x = np.random.randn(5, 6)

    d = layer.derivative(x)

    assert np.all(d >= 0.0)
    assert np.all(d <= 0.25 + 1e-12)


def test_softmax_backward_uses_elementwise_derivative():
    layer = SoftmaxActivation()
    x = np.random.randn(3, 5)

    _ = layer.forward_propagation(x, training=True)

    output_error = np.ones_like(x)
    grad_in = layer.backward_propagation(output_error)

    s = layer.activation_function(x)
    expected = s * (1.0 - s)
    assert np.allclose(grad_in, expected, atol=1e-12)


def test_softmax_output_shape_and_parameters():
    layer = SoftmaxActivation()
    x = np.random.randn(2, 4)

    y = layer.forward_propagation(x, training=True)

    assert y.shape == x.shape
    assert layer.parameters() == 0
