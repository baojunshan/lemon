from lemon.base import BaseClassifierModel
import numpy as np


class LinearRegression(BaseClassifierModel):
    def __init__(self):
        self.w = None

    @staticmethod
    def _add_b(x):
        return np.concatenate((x, np.ones(x.shape[0]).reshape(-1, 1)), axis=1)

    @staticmethod
    def _lsm(x, y):
        return np.linalg.inv(x.T @ x) @ x.T @ y

    def fit(self, x, y):
        self.w = self._lsm(self._add_b(self._to_numpy(x)), self._to_numpy(y))
        return self

    def predict(self, x):
        return self._add_b(self._to_numpy(x)) @ self.w

    @staticmethod
    def score(y_true, y_pred):
        # FIXME: need implement r2_score
        # return r2_score(y_true, y_pred)
        return None


class LinearRegressionGD(BaseClassifierModel):
    def __init__(self, max_iter=10000, lr=0.01):
        self.max_iter = max_iter
        self.lr = lr
        self.w = None
        self.b = None

    def _sgd(self, x, y):
        w = np.zeros([x.shape[1]])
        b = 0
        for i in range(self.max_iter):
            p = x @ w + b
            w -= np.dot(x.T, p - y) / x.shape[0] * self.lr
            b -= np.sum((p - y)) / x.shape[0] * self.lr
        return w, b

    def fit(self, x, y):
        self.w, self.b = self._sgd(self._to_numpy(x), self._to_numpy(y))
        return self

    def predict(self, x):
        return self._to_numpy(x) @ self.w + self.b


if __name__ == "__main__":
    X = np.array([[1.1, 1], [1.2, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    print(y)
    model = LinearRegression().fit(X, y)
    # model.score(X, y)
    print(model.predict(X))

    model_gd = LinearRegressionGD().fit(X, y)
    print(model_gd.predict(X))
