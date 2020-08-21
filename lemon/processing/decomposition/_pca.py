from ...base import BaseDecompositor


class PCA(BaseDecompositor):
    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvectors = None

    def fit(self, X):
        m = X.shape[0]
        X = X - np.mean(X, axis=0)
        cov = 1 / m * (X.T @ X)

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        idx = np.argsort(eigenvalues[::-1])
        eigenvectors = eigenvectors[:, idx]
        self.eigenvectors = eigenvectors[:, :self.n_components]
        return self

    def transform(self, X):
        return self.eigenvectors @ X

    def fit_transform(self, X):
        return self.fit(X).transform(X)