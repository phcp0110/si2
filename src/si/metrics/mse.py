import numpy as np
from si.metrics.mse import mse

class RidgeRegressionLeastSquares:
    def __init__(self, l2_penalty: float = 1.0, scale: bool = True):
        """
        Initialize the RidgeRegressionLeastSquares model.

        Parameters:
        -----------
        l2_penalty: float, default=1.0
            L2 regularization parameter.
        scale: bool, default=True
            Whether to scale the data or not.
        """
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Estimate the theta and theta_zero coefficients, mean, and std.

        Parameters:
        -----------
        X: np.ndarray
            The input features.
        y: np.ndarray
            The target values.
        """
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        m, n = X.shape
        X_with_intercept = np.c_[np.ones(m), X]

        # Penalty matrix
        penalty_matrix = self.l2_penalty * np.eye(n + 1)
        penalty_matrix[0, 0] = 0

        # Compute theta
        theta = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept) + penalty_matrix).dot(X_with_intercept.T).dot(y)
        self.theta_zero = theta[0]
        self.theta = theta[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the dependent variable (y) using the estimated theta coefficients.

        Parameters:
        -----------
        X: np.ndarray
            The input features.

        Returns:
        --------
        np.ndarray
            The predicted values.
        """
        if self.scale:
            X = (X - self.mean) / self.std
        X_with_intercept = np.c_[np.ones(len(X)), X]
        return X_with_intercept.dot(np.r_[self.theta_zero, self.theta])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the error between the real and predicted y values.

        Parameters:
        -----------
        X: np.ndarray
            The input features.
        y: np.ndarray
            The true target values.

        Returns:
        --------
        float
            The mean squared error.
        """
        y_pred = self.predict(X)
        return mse(y, y_pred)
