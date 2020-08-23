from ...base import BaseDecompositor

import numpy as np


class PCA(BaseDecompositor):
    def __init__(self, n_components):
        self.n_components = n_components
        self.eigen_vectors = None

    def fit(self, X):
        m = X.shape[0]
        X = X - np.mean(X, axis=0)
        cov = 1 / m * (X.T @ X)

        eigen_values, eigen_vectors = np.linalg.eig(cov)

        idx = np.argsort(eigen_values[::-1])
        eigen_vectors = eigen_vectors[:, idx]
        self.eigen_vectors = eigen_vectors[:, :self.n_components]
        return self

    def transform(self, X):
        return self.eigen_vectors @ X

    def fit_transform(self, X):
        return self.fit(X).transform(X)