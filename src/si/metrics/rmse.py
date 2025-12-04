import numpy as np
from si.metrics.mse import mse


def rmse(y_true, y_pred) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    float
        RMSE value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return float(np.sqrt(mse(y_true, y_pred)))
