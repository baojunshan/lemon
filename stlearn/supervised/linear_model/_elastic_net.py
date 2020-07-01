from stlearn.base import BaseClassifierModel
import numpy as np


class ElasticNetGD(BaseClassifierModel):
    def __init__(self, lr=0.01, alpha=0.4, max_iter=10000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.lr = lr
        self.w = None
        self.b = None

    @staticmethod
    def _l1_grad(w):
        w[w > 0] = 1
        w[2 < 0] = -1
        return w

    def _sgd(self, x, y):
        w, b = np.zeros([x.shape[1]]), 0
        for i in range(self.max_iter):
            p = x @ w + b
            w -= (x.T @ (p - y) / x.shape[0]
                  + self.alpha * self._l1_grad(w)
                  + (1 - self.alpha) * self.w) * self.lr
            b -= np.sum((p - y)) / x.shape[0] * self.lr
        return w, b

    def fit(self, x, y):
        self.w, self.b = self._sgd(self._to_numpy(x), self._to_numpy(y))
        return self

    def predict(self, x):
        x = self._to_numpy(x)
        return self._to_numpy(x) @ self.w + self.b


if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([2, 3, 4, 5])
    print(y)
    model = ElasticNetGD(alpha=0.4).fit(X, y)
    print(model.predict(X))
    print(model.w)
