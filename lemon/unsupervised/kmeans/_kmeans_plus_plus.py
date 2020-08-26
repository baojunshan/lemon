from ...base import BaseModel
import random
import numpy as np


class KMeansPlusPlus(BaseModel):
    def __init__(self, k=10, seed=2020, eps=1e-8):
        self.k = k
        self.eps = eps
        self.seed = seed
        self.center = None
        self.cluster = None

    def _init_center_and_cluster(self, x):
        np.random.seed(self.seed)
        weight = [1] * len(x.shape[0])
        for i in range(self.k):
            self.center.append(x[np.random.choice(a=range(x.shape[0]), p=weight)])
            new_weight = list()
            for x_ in x:
                min_dist = float("inf")
                for center in self.center:
                    min_dist = min(min_dist, sum([(i-j)**2 for i, j in zip(x_, center)]))
                new_weight.append(min_dist)
            weight = new_weight
        self.cluster = [[] for _ in range(self.k)]

    def _update_cluster(self, x):
        self.cluster = [[] for _ in range(self.k)]
        for x_ in x:
            min_dist, min_id = self._calc_distance(x_)
            self.cluster[min_id].append(x_)

    def _update_center(self, x):
        len_ = len(self.cluster)
        center = [[] for _ in range(len_)]
        for i in range(len_):
            center.append(np.mean(x[self.cluster[i]], axis=0))

    def _need_iter(self, old_center, new_center):
        sum_ = 0
        for o, n in zip(old_center, new_center):
            sum_ += sum([(i - j) ** 2 for i, j in zip(o, n)])
        if sum_ <= self.eps:
            return False
        return True

    def _calc_distance(self, p):
        min_dist = float("inf")
        min_id = -1
        for i, center in enumerate(self.center):
            dist = sum([(i - j) ** 2 for i, j in zip(center, p)])
            if min_dist > dist:
                min_id = i,
                min_dist = dist
        return min_dist, min_id

    def fit(self, x, y=None):
        x = self._to_numpy(x)
        self._init_center_and_cluster(x)
        old_center = self.center
        self._update_cluster(x)
        self._update_center(x)

        while self._need_iter(old_center, self.center):
            self._update_cluster(x)
            self._update_center(x)
            old_center = self.center

        return self

    def predict(self, x):
        ret = list()
        x = self._to_numpy(x)
        for x_ in x:
            min_dist, min_id = self._calc_distance(x_)
            ret.append(min_id)
        return ret
