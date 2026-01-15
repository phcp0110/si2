import numpy as np
import pytest

from si.neural_networks.optimizers import Adam


def test_adam_initializes_state_on_first_update():
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    w = np.array([1.0, 2.0, 3.0])
    g = np.array([0.1, -0.2, 0.3])

    w2 = opt.update(w, g)

    assert opt.m is not None
    assert opt.v is not None
    assert opt.t == 1
    assert opt.m.shape == w.shape
    assert opt.v.shape == w.shape
    assert w2.shape == w.shape


def test_adam_does_not_modify_w_in_place():
    opt = Adam()
    w = np.array([1.0, 2.0, 3.0])
    w_copy = w.copy()
    g = np.array([0.1, 0.1, 0.1])

    _ = opt.update(w, g)
    assert np.array_equal(w, w_copy)


def test_adam_zero_gradient_keeps_weights_constant():
    opt = Adam(learning_rate=0.01)

    w = np.array([1.0, -2.0, 0.5])
    g = np.zeros_like(w)

    w2 = opt.update(w, g)
    assert np.allclose(w2, w, atol=1e-15)


def test_adam_first_step_matches_closed_form_for_constant_grad():
    """
    For first step with m0=v0=0:
      m1 = (1-beta1)*g
      v1 = (1-beta2)*g^2
      m_hat = m1/(1-beta1) = g
      v_hat = v1/(1-beta2) = g^2
      w1 = w0 - lr * g / (|g| + eps)
    Elementwise.
    """
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    opt = Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=eps)

    w0 = np.array([1.0, 2.0, 3.0], dtype=float)
    g = np.array([0.1, -0.2, 0.3], dtype=float)

    w1 = opt.update(w0, g)

    expected = w0 - lr * (g / (np.abs(g) + eps))
    assert np.allclose(w1, expected, atol=1e-12)


def test_adam_t_increments_each_update():
    opt = Adam()
    w = np.array([1.0, 2.0])
    g = np.array([0.1, 0.1])

    _ = opt.update(w, g)
    assert opt.t == 1
    _ = opt.update(w, g)
    assert opt.t == 2
    _ = opt.update(w, g)
    assert opt.t == 3


def test_adam_state_updates_match_recurrence():
    """
    Check m and v updates after 2 steps against recurrence definitions.
    """
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    opt = Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=eps)

    w = np.array([1.0, 1.0], dtype=float)
    g1 = np.array([0.2, -0.4], dtype=float)
    g2 = np.array([0.1, 0.3], dtype=float)

    _ = opt.update(w, g1)
    m1_expected = (1 - beta1) * g1
    v1_expected = (1 - beta2) * (g1 ** 2)
    assert np.allclose(opt.m, m1_expected, atol=1e-12)
    assert np.allclose(opt.v, v1_expected, atol=1e-12)

    _ = opt.update(w, g2)
    m2_expected = beta1 * m1_expected + (1 - beta1) * g2
    v2_expected = beta2 * v1_expected + (1 - beta2) * (g2 ** 2)
    assert np.allclose(opt.m, m2_expected, atol=1e-12)
    assert np.allclose(opt.v, v2_expected, atol=1e-12)


def test_adam_updates_finite_for_large_gradients():
    opt = Adam(learning_rate=0.001)

    w = np.array([1.0, -1.0, 0.5], dtype=float)
    g = np.array([1e6, -1e6, 1e8], dtype=float)

    w2 = opt.update(w, g)
    assert np.isfinite(w2).all()


def test_adam_accepts_list_inputs_and_returns_numpy_array():
    opt = Adam()

    w = [1.0, 2.0, 3.0]
    g = [0.1, 0.1, 0.1]

    w2 = opt.update(w, g)
    assert isinstance(w2, np.ndarray)
    assert w2.shape == (3,)
