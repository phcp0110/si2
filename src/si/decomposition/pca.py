import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calculate the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Infer the principal components
        self.components = eigenvectors[:, :self.n_components]

        # Infer the explained variance
        self.explained_variance = eigenvalues[:self.n_components] / np.sum(eigenvalues)

    def _transform(self, X):
        # Center the data
        X_centered = X - self.mean

        # Calculate the reduced X
        X_reduced = np.dot(X_centered, self.components)

        return X_reduced
