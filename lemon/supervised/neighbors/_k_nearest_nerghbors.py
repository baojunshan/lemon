from lemon.base import BaseModel
from lemon.supervised.neighbors._kd_tree import KdTree
import numpy as np
import heapq
from collections import Counter


class KNearestNeighbors(BaseModel):
    def __init__(self, n_neighbors=20, algorithm="auto",
                 leaf_size=30, distance_type=2, mode="classifier"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.distance_type = distance_type
        self.mode = mode

        self.x_ = None
        self.y_ = None
        self.tree_ = None

    def _update_algorithm(self, x):
        if self.algorithm not in ('auto', 'brute', 'kd_tree'):
            raise ValueError("Algorithm should be 'auto', 'brute' or 'kd_tree'.")
        if self.algorithm == "auto":
            if x.shape[0] > 1000:
                self.algorithm = "kd_tree"
            else:
                self.algorithm = "brute"

    def _brute_search(self, x):
        distances = None
        if isinstance(self.distance_type, int):
            distances = np.linalg.norm(self.x_ - x, ord=2, axis=0)
        if self.distance_type == "inf":
            distances = np.apply_along_axis(lambda array: chebyshev_distance(array, x),
                                            axis=1, arr=self.x_)
        sort_ret = heapq.nsmallest(n=self.n_neighbors, iterable=zip(distances, self.y_))
        sort_ret = [v[1] for v in sort_ret]
        if self.mode == "classifier":
            return Counter(sort_ret).most_common(1)[0][0]
        else:
            return sum(sort_ret) / len(sort_ret)

    def _kd_tree_search(self, x):
        ret = [v[1] for v in self.tree_.search(point=x, k=self.n_neighbors)]
        ret = [self.y_[np.where((self.x_ == p).all(1))[0][0]] for p in ret]
        if self.mode == "classifier":
            return Counter(ret).most_common(1)[0][0]
        else:
            return sum(ret) / len(ret)

    def fit(self, x, y):
        self._update_algorithm(x)
        self.x_ = self._to_numpy(x)
        self.y_ = self._to_numpy(y)
        if self.algorithm == "brute":
            pass
        elif self.algorithm == "kd_tree":
            self.tree_ = KdTree(self.x_, leaf_size=self.leaf_size)
        return self

    def predict(self, x):
        x = self._to_numpy(x)
        if self.algorithm == "brute":
            return np.array([self._brute_search(v) for v in x])
        elif self.algorithm == "kd_tree":
            return np.array([self._kd_tree_search(v) for v in x])
        return


class KNearestNeighborsClassifier(KNearestNeighbors):
    def __init__(self, n_neighbors=20, algorithm="auto",
                 leaf_size=30, distance_type=2):
        super(KNearestNeighborsClassifier, self).__init__(n_neighbors=n_neighbors,
                                                          algorithm=algorithm,
                                                          leaf_size=leaf_size,
                                                          distance_type=distance_type,
                                                          mode="classifier")


class KNearestNeighborsRegression(KNearestNeighbors):
    def __init__(self, n_neighbors=20, algorithm="auto",
                 leaf_size=30, distance_type=2):
        super(KNearestNeighborsRegression, self).__init__(n_neighbors=n_neighbors,
                                                          algorithm=algorithm,
                                                          leaf_size=leaf_size,
                                                          distance_type=distance_type,
                                                          mode="regression")


if __name__ == "__main__":
    x = np.array([[1, 2], [1, 2], [2, 2], [4, 10]])
    y = np.array([1, 1, 2, 5])
    model = KNearestNeighbors(n_neighbors=2, algorithm="kd_tree", mode="regression").fit(x, y)
    print(model.predict([[1, 1]]))
