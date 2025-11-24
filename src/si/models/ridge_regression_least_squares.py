import numpy as np
from si.metrics.mse import mse

class RidgeRegression:
    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.01, max_iter: int = 1000, patience: int = 10, scale: bool = True):
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        m, n = X.shape
        self.theta = np.zeros(n)
        self.theta_zero = 0

        prev_cost = float('inf')
        patience_counter = 0

        for iteration in range(self.max_iter):
            y_pred = self._predict(X)

            # Update theta
            for j in range(n):
                gradient = (1/m) * np.sum((y_pred - y) * X[:, j])
                self.theta[j] = (1 - self.alpha * self.l2_penalty / m) * self.theta[j] - self.alpha * gradient

            # Update theta_zero
            gradient_zero = (1/m) * np.sum(y_pred - y)
            self.theta_zero -= self.alpha * gradient_zero

            # Compute cost
            cost = self.cost(X, y)
            self.cost_history[iteration] = cost

            # Early stopping
            if abs(prev_cost - cost) < 1e-6:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
            else:
                patience_counter = 0
            prev_cost = cost

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.scale:
            X = (X - self.mean) / self.std
        return self.theta_zero + np.dot(X, self.theta)

    def cost(self, X: np.ndarray, y: np.ndarray) -> float:
        m = len(y)
        y_pred = self._predict(X)
        error = y_pred - y
        regularization = self.l2_penalty * np.sum(self.theta ** 2)
        return (1 / (2 * m)) * (np.sum(error ** 2) + regularization)

    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self._predict(X)
        return mse(y, y_pred)
