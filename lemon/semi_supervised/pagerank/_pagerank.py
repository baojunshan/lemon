from ...base import BaseModel
import numpy as np


class PageRank(BaseModel):
    def __init__(self, eps=1e-5, d=0.85, max_iter=1000):
        self.eps = eps
        self.d = d
        self.max_iter = max_iter

    @staticmethod
    def calc_bias(old, new):
        return np.sqrt(sum([(o-n)**2 for o, n in zip(old, new)]))

    def fit(self, x, y=None):
        return self

    def predict(self, x):
        ps = set()
        for x_ in x:
            ps.add(x_[0])
            ps.add(x_[1])
        p_num = len(ps)
        A = np.zeros(p_num, p_num)

        for x_ in x:
            weight = 1. if len(x_) < 3 else x_[2]
            A[x_[0], x_[1]] += weight
        A = A / np.sum(A, axis=0)
        w = np.array([1 / p_num] * p_num)

        old_w = w
        b = (1 - self.d) / p_num
        for _ in range(self.max_iter):
            w = A @ w * self.d + b
            if self.calc_bias(old_w, w) < self.eps:
                break
            old_w = w
        return {k: v for k, v in zip(range(p_num), w)}
