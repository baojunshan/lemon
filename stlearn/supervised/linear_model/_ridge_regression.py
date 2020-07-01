from stlearn.base import BaseClassifierModel
import numpy as np


class RidgeRegression(BaseClassifierModel):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.w = None

    @staticmethod
    def _add_b(x):
        return np.concatenate((x, np.ones(x.shape[0]).reshape(-1, 1)), axis=1)

    def _lsm(self, x, y):
        w = np.linalg.inv(x.T @ x + np.eye(x.shape[1]) * self.alpha) @ x.T @ y
        return w

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


class RidgeRegressionGD(BaseClassifierModel):
    def __init__(self, max_iter=10000, lr=0.01, alpha=0.1):
        self.max_iter = max_iter
        self.lr = lr
        self.alpha = alpha
        self.w = None
        self.b = None

    def _sgd(self, x, y):
        w, b = np.zeros([x.shape[1]]), 0
        for i in range(self.max_iter):
            p = x @ w + b
            w -= (x.T @ (p - y) / x.shape[0] + 2 * self.alpha * w) * self.lr
            b -= np.sum((p - y)) / x.shape[0] * self.lr
        return w, b

    def fit(self, x, y):
        self.w, self.b = self._sgd(self._to_numpy(x), self._to_numpy(y))
        return self

    def predict(self, x):
        x = self._to_numpy(x)
        return self._to_numpy(x) @ self.w + self.b

    @staticmethod
    def score(y_true, y_pred):
        # FIXME: need implement r2_score
        # return r2_score(y_true, y_pred)
        return None


if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([2, 3, 4, 5])
    print(y)
    model = RidgeRegression(alpha=0.01).fit(X, y)
    # model.score(X, y)
    print(model.predict(X))

    model_gd = RidgeRegressionGD(alpha=0.01).fit(X, y)
    print(model_gd.predict(X))

