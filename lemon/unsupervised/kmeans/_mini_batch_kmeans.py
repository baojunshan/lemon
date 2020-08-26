from ...base import BaseModel
import random
import numpy as np


class MiniBatchKMeans(BaseModel):
    def __init__(self, k=10, seed=2020, eps=1e-8, batch_size=1e3):
        self.k = k
        self.eps = eps
        self.seed = seed
        self.batch_size = int(batch_size)
        self.center = None
        self.cluster = None

    def _init_center_and_cluster(self, x):
        random.seed(self.seed)
        for i in range(self.k):
            self.center.append(x[random.choice(range(x.shape[0]))])
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
            sum_ += sum([(i-j)**2 for i, j in zip(o, n)])
        if sum_ <= self.eps:
            return False
        return True

    def _calc_distance(self, p):
        min_dist = float("inf")
        min_id = -1
        for i, center in enumerate(self.center):
            dist = sum([(i-j)**2 for i, j in zip(center, p)])
            if min_dist > dist:
                min_id = i,
                min_dist = dist
        return min_dist, min_id

    def _next_batch(self, x):
        len_ = len(x.shape[0])
        n = len_ // self.batch_size
        while True:
            for i in range(n):
                yield x[i*self.batch_size, (i+1)*self.batch_size]

    def fit(self, x, y=None):
        x = self._to_numpy(x)
        x_iter = self._next_batch(x)
        curr_x = next(x_iter)
        self._init_center_and_cluster(curr_x)
        old_center = self.center
        self._update_cluster(curr_x)
        self._update_center(curr_x)

        while self._need_iter(old_center, self.center):
            self._update_cluster(curr_x)
            self._update_center(curr_x)
            old_center = self.center
            curr_x = next(x_iter)
        return self

    def predict(self, x):
        ret = list()
        x = self._to_numpy(x)
        for x_ in x:
            min_dist, min_id = self._calc_distance(x_)
            ret.append(min_id)
        return ret
