"""
optimizers.py
-------------
Optimizers for Neural Networks.

Per slides:
- Create a base Optimizer class with:
  - learning_rate
  - update(w, grad_loss_w)

Implement Adam (Exercise 15):
- parameters: learning_rate, beta_1, beta_2, epsilon
- estimated params: m, v, t
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    Base optimizer class.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = float(learning_rate)

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update parameters (weights or biases) given the gradient.
        """
        raise NotImplementedError


class Adam(Optimizer):
    """
    Adam optimizer.

    Combines momentum (1st moment) and RMS scaling (2nd moment).

    Parameters
    ----------
    learning_rate : float
        Step size.
    beta_1 : float
        Exponential decay for 1st moment (default 0.9).
    beta_2 : float
        Exponential decay for 2nd moment (default 0.999).
    epsilon : float
        Numerical stability constant (default 1e-8).

    Estimated Parameters
    --------------------
    m : np.ndarray | None
        Moving average of gradients (1st moment).
    v : np.ndarray | None
        Moving average of squared gradients (2nd moment).
    t : int
        Time step (epoch/iteration counter).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate=learning_rate)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)

        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Adam update step.

        Algorithm (per slides):
        1) initialize m, v to zeros if needed
        2) t += 1
        3) m = beta1*m + (1-beta1)*grad
        4) v = beta2*v + (1-beta2)*grad^2
        5) bias correction:
           m_hat = m / (1 - beta1^t)
           v_hat = v / (1 - beta2^t)
        6) update:
           w = w - lr * m_hat / (sqrt(v_hat) + eps)
        """
        w = np.asarray(w, dtype=float)
        grad = np.asarray(grad_loss_w, dtype=float)

        if self.m is None:
            self.m = np.zeros_like(w)
        if self.v is None:
            self.v = np.zeros_like(w)

        self.t += 1

        # 1st moment (momentum)
        self.m = self.beta_1 * self.m + (1.0 - self.beta_1) * grad

        # 2nd moment (RMS)
        self.v = self.beta_2 * self.v + (1.0 - self.beta_2) * (grad ** 2)

        # Bias correction
        m_hat = self.m / (1.0 - (self.beta_1 ** self.t))
        v_hat = self.v / (1.0 - (self.beta_2 ** self.t))

        # Update weights
        w_updated = w - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))
        return w_updated
