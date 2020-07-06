from lemon.base import BaseModel
import numpy as np


class LogisticRegression(BaseModel):
    def __init__(self, solver="gd", lr=0.01, max_iter=10000):
        self.solver = solver  # "gd", "newton-cg", "lbfgs"
        self.lr = lr
        self.max_iter = max_iter
        self.w = None

    @staticmethod
    def _add_b(x):
        return np.concatenate((x, np.ones(x.shape[0]).reshape(-1, 1)), axis=1)

    @staticmethod
    def _sigmoid(x, w):
        exp = np.exp(x @ w)
        return exp / (1 + exp)

    def _gd(self, x, y):
        for i in range(self.max_iter):
            grad = (1 / x.shape[0]) * (y - self._sigmoid(x, self.w)) @ x
            self.w += self.lr * grad
        return self.w

    def _newton_cg(self, x, y):
        # TODO
        print("it will be updated in the future~")
        return self.w

    def _lbfgs(self, x, y):
        # TODO
        print("it will be updated in the future~")
        return self.w

    def fit(self, x, y):
        x = self._add_b(self._to_numpy(x))
        y = self._to_numpy(y)
        self.w = np.zeros([x.shape[1]])
        if self.solver == "gd":
            self.w = self._gd(x, y)
        elif self.solver == "newton-cg":
            self.w = self._newton_cg(x, y)
        elif self.solver == "lbfgs":
            self.w = self._lbfgs(x, y)
        else:
            raise ValueError("sovler should be 'gd', 'newton-cg' or 'lbfgs'!")
        return self

    def predict(self, x):
        x = self._add_b(self._to_numpy(x))
        pred = self._sigmoid(x, self.w)
        return np.where(pred >= 0.5, 1, 0)
