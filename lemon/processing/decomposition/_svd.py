from ...base import BaseDecompositor

import numpy as np


class SVD(BaseDecompositor):
    def __init__(self, n_components):
        self.n_components = n_components
        self.U = None
        self.V = None
        self.sigma = None

    def fit(self, X):
        U_values, U = np.linalg.eig(X @ X.T)
        V_values, V = np.linalg.eig(X.T @ X)
        sigma = np.sqrt(V_values)

        self.U = U[:, :self.n_components]
        self.V = V[:self.n_components, :]
        self.sigma = sigma[:self.n_components]
        return self

    def transform(self, X):
        return self.V @ X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
