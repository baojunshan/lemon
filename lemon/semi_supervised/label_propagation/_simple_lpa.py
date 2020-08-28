from ...base import BaseModel

import numpy as np


class SimpleLPA(BaseModel):
    def __init__(self, max_iter=1000, random_state=2020):
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _is_same(old, new):
        for o, n in zip(old, new):
            if o != n:
                return False
        return True

    def fit(self, x=None, y=None):
        return self

    def predict(self, x):
        np.random.seed(self.random_state)
        neighbors = dict()
        for x_ in x:
            if x_[0] not in neighbors.keys():
                neighbors[x_[0]] = dict()
            if x_[1] not in neighbors[x_[0]].keys():
                neighbors[x_[0]][x_[1]] = 0
            neighbors[x_[0]][x_[1]] += 1. if len(x_) < 3 else x_[2]

            if x_[1] not in neighbors.keys():
                neighbors[x_[1]] = dict()
            if x_[0] not in neighbors[x_[1]].keys():
                neighbors[x_[1]][x_[0]] = 0
            neighbors[x_[1]][x_[0]] += 1. if len(x_) < 3 else x_[2]

        label = [i for i in range(len(neighbors.keys()))]
        for _ in range(self.max_iter):
            old_label = label
            for id in neighbors.keys():
                a, p = list(), list()
                for i, w in neighbors[id].items():
                    a.append(old_label[i])
                    p.append(w)
                label[id] = np.random.choice(a, p)
            if self._is_same(old_label, label):
                break
        return label








