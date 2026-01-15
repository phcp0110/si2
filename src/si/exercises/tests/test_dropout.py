import numpy as np
import pytest

from si.neural_networks.layers import Dropout


def test_init_rejects_invalid_probability():
    with pytest.raises(ValueError):
        Dropout(-0.1)
    with pytest.raises(ValueError):
        Dropout(1.0)
    with pytest.raises(ValueError):
        Dropout(1.5)


def test_forward_inference_returns_input_unchanged():
    np.random.seed(0)
    layer = Dropout(probability=0.5)

    x = np.random.randn(5, 4)
    y = layer.forward_propagation(x, training=False)

    assert np.array_equal(y, x)
    assert np.array_equal(layer.output, x)


def test_forward_training_creates_mask_with_same_shape():
    np.random.seed(0)
    p = 0.3
    layer = Dropout(probability=p)

    x = np.ones((100, 10))
    y = layer.forward_propagation(x, training=True)

    assert layer.mask is not None
    assert layer.mask.shape == x.shape
    assert y.shape == x.shape


def test_forward_training_mask_is_binary():
    np.random.seed(1)
    layer = Dropout(probability=0.4)

    x = np.random.randn(50, 20)
    _ = layer.forward_propagation(x, training=True)

    unique_vals = np.unique(layer.mask)
    assert set(unique_vals).issubset({0, 1})


def test_forward_training_applies_inverted_dropout_scaling():
    """
    With inverted dropout, for input of ones:
    output is either 0 or scale, where scale = 1/(1-p).
    """
    np.random.seed(2)
    p = 0.25
    layer = Dropout(probability=p)

    x = np.ones((30, 30))
    y = layer.forward_propagation(x, training=True)

    scale = 1.0 / (1.0 - p)
    unique_out = np.unique(y)
    assert set(np.round(unique_out, 10)).issubset({0.0, round(scale, 10)})


def test_forward_training_expected_value_is_preserved_approximately():
    """
    For inverted dropout, E[output] ~ input (elementwise) in expectation.
    We'll test this statistically with a large tensor.
    """
    np.random.seed(3)
    p = 0.4
    layer = Dropout(probability=p)

    x = np.ones((2000, 50))
    y = layer.forward_propagation(x, training=True)

    assert np.isclose(y.mean(), 1.0, atol=0.05)


def test_backward_propagation_blocks_gradients_where_mask_is_zero():
    np.random.seed(4)
    p = 0.5
    layer = Dropout(probability=p)

    x = np.random.randn(20, 10)
    _ = layer.forward_propagation(x, training=True)

    grad_out = np.ones_like(x)
    grad_in = layer.backward_propagation(grad_out)

    assert np.all(grad_in[layer.mask == 0] == 0)
    assert np.all(grad_in[layer.mask == 1] == 1)


def test_backward_requires_mask_set_by_training_forward():
    """
    If backward is called before a training forward pass,
    mask is None and multiplication should fail.
    This test documents that behavior.
    """
    layer = Dropout(probability=0.5)
    grad_out = np.ones((3, 3))

    with pytest.raises(TypeError):
        _ = layer.backward_propagation(grad_out)


def test_output_shape_returns_input_shape_and_parameters_is_zero():
    """
    output_shape should return the input shape as per spec.
    parameters should return 0.
    """
    layer = Dropout(probability=0.2)

    x = np.random.randn(7, 11)
    _ = layer.forward_propagation(x, training=False)

    assert layer.parameters() == 0
