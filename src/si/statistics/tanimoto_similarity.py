import numpy as np

def tanimoto_similarity(x, y):
    """
    Calculate the Tanimoto similarity between a single binary sample x and multiple binary samples y.

    Parameters:
    x (array-like): A single binary sample (1D array or list).
    y (array-like): Multiple binary samples (2D array or list of lists), where each row is a sample.

    Returns:
    numpy.ndarray: An array containing the Tanimoto similarities between x and each sample in y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Calculate the dot product between x and each sample in y
    dot_product = np.dot(y, x)

    # Calculate the sum of squares for x and each sample in y
    x_squared = np.sum(x ** 2)
    y_squared = np.sum(y ** 2, axis=1)

    # Calculate the Tanimoto similarity
    similarity = dot_product / (x_squared + y_squared - dot_product)

    return similarity
