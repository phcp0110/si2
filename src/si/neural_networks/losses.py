"""
losses.py
---------
Loss functions for Neural Networks.

Per slides:
- Create a base LossFunction class with two abstract methods:
  - loss(y_true, y_pred)
  - derivative(y_true, y_pred)

Then implement specific losses, including:
- CategoricalCrossEntropy (Exercise 14)

Notes:
- Use np.clip to avoid log(0) and division by 0 (explicitly requested).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    """
    Base class for loss functions.
    """

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the loss with respect to predictions y_pred.
        """
        raise NotImplementedError


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross-Entropy loss for multi-class classification.

    Assumptions:
    - y_true is one-hot encoded: shape (n_samples, n_classes)
    - y_pred are probabilities (typically after softmax): same shape

    Formula (average over samples):
        L = - mean( sum_c y_true[c] * log(y_pred[c]) )

    Derivative (element-wise form used in the slides):
        dL/dy_pred = - y_true / y_pred

    Numerical stability:
    - clip y_pred to avoid log(0) and division by 0.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape (one-hot vs probabilities).")

        # Avoid log(0)
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        # Compute per-sample losses then average
        per_sample = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return float(np.mean(per_sample))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape (one-hot vs probabilities).")

        # Avoid division by 0
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        return -y_true / y_pred_clipped
