from ...base import BaseModel

import numpy as np


class DBSCAN(BaseModel):
    def __init__(self, eta=1, min_pts=5, random_seed=2020):
        self.eta = eta
        self.min_pts = min_pts
        self.core_objs = None
        self.c = None
        self.use_points = None
        self.p2c = None
        self.k = 0
        self.random_seed = random_seed

    @staticmethod
    def _dist(p1, p2):
        return sum([(i - j) ** 2 for i, j in zip(p1, p2)]) ** 0.5

    def _get_neighbor_ids(self, x, p):
        res = list()
        for i in range(x.shape[0]):
            if self._dist(p, x[i]) < e:
                res.append(i)
        return res

    def fit(self, x, y=None):
        x = self._to_numpy(x)
        self.core_objs = dict()
        self.c = dict()
        self.k = 0
        np.random.seed(self.random_seed)

        for i in range(x.shape[0]):
            neighbor_ids = self._get_neighbor_ids(x, x[i])
            if len(neighbor_ids) >= self.min_pts:
                self.core_objs[i] = neighbor_ids

        old_core_objs = self.core_objs.copy()
        not_access = list(range(x.shape[0]))
        while len(self.core_objs) > 0:
            old_not_access = list()
            old_not_access.extend(not_access)
            cores = list(self.core_objs.keys())
            core = cores[np.random.randint(0, len(cores))]

            queue = list()
            queue.append(core)
            not_access.remove(core)
            while len(queue) > 0:
                q = queue.pop()
                if q in old_core_objs.keys():
                    delte = [val for val in old_core_objs[q] if val in not_access]
                    queue.extend(delte)
                    not_access = [val for val in not_access if val not in delte]
            self.k += 1
            self.c[self.k] = [val for val in old_not_access if val not in not_access]
            for i in self.c[self.k]:
                if i in self.core_objs.keys():
                    del self.core_objs[i]

        self.use_points = dict()
        self.p2c = dict()
        for k in range(1, self.k + 1):
            for i in self.c[k]:
                self.use_points[i] = x[i]
                self.p2c[i] = k

        return self

    def predict(self, x):
        x = self._to_numpy(x)
        ret = list()
        for x_ in x:
            pts_id = list()
            for i in self.use_points.keys():
                dist = self._dist(x_, self.use_points[i])
                if dist < self.eta:
                    pts_id.append(i)
            if len(pts_id) >= self.min_pts:
                c_dict = dict()
                for c in [self.p2c[p] for p in pts_id]:
                    if c not in c_dict.keys():
                        c_dict[c] = 0
                    c_dict[c] += 1
                cluster_id = sorted(c_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
                ret.append(cluster_id)
            else:
                ret.append(-1)
        return ret
