import numpy as np
from si.statistics.euclidean_distance import euclidean_distance

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset.
    y_pred: np.ndarray
        The predicted labels of the dataset.

    Returns
    -------
    rmse: float
        The root mean squared error of the model.
    """
    mse = np.sum((y_true - y_pred) ** 2) / len(y_true)
    return np.sqrt(mse)

class KNNRegressor:
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Store the training data.

        Parameters
        ----------
        X_train: np.ndarray
            The training features.
        y_train: np.ndarray
            The training labels.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the test data.

        Parameters
        ----------
        X_test: np.ndarray
            The test features.

        Returns
        -------
        y_pred: np.ndarray
            The predicted labels for the test data.
        """
        y_pred = np.zeros(X_test.shape[0])

        for i, x_test in enumerate(X_test):
            # Calculate Euclidean distances using your function
            distances = euclidean_distance(x_test, self.X_train)

            # Get the indices of the k nearest neighbors
            k_indices = np.argpartition(distances, self.k)[:self.k]

            # Get the corresponding y values
            k_nearest_labels = self.y_train[k_indices]

            # Calculate the average of the k nearest labels
            y_pred[i] = np.mean(k_nearest_labels)

        return y_pred

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculate the RMSE score for the test data.

        Parameters
        ----------
        X_test: np.ndarray
            The test features.
        y_test: np.ndarray
            The true labels for the test data.

        Returns
        -------
        rmse: float
            The RMSE score.
        """
        y_pred = self.predict(X_test)
        return rmse(y_test, y_pred)
