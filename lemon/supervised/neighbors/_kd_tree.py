from lemon.base import _to_numpy
import numpy as np
import itertools
import heapq
import math


class KdNode:
    def __init__(self, value=None, split_axis=None, parent=None, left=None, right=None):
        self.value = value
        self.split_axis = split_axis
        self.left = left
        self.right = right
        self.parent = parent

    @property
    def is_leaf(self):
        return self.value.ndim == 2

    @property
    def children(self):
        if self.left:
            yield self.left
        if self.right:
            yield self.right

    @property
    def depth(self):
        min_height = int(bool(self))
        return max([min_height] + [c.height() + 1 for c, p in self.children])

    def pre_order(self):
        if not self:
            return
        yield self
        if self.left:
            for x in self.left.preorder():
                yield x
        if self.right:
            for x in self.right.preorder():
                yield x

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.value == other
        else:
            return self.value == other.data

    def _search_node(self, point, k, results, get_dist, counter):
        if not self:
            return
        split_axis = self.split_axis if self.split_axis is not None else 0
        if self.is_leaf:
            self_vs = [p for p in self.value]
        else:
            self_vs = [self.value]

        curr_value = self_vs[0]
        curr_dist = -math.inf
        for v in self_vs:
            node_dist = get_dist(v, point)
            item = (-node_dist, v.tolist(), next(counter), self)
            if len(results) >= k:
                if -node_dist > results[0][0]:
                    heapq.heapreplace(results, item)
                    if curr_dist < node_dist:
                        curr_value = v
                        curr_dist = node_dist
            else:
                heapq.heappush(results, item)
                if curr_dist < node_dist:
                    curr_value = v
                    curr_dist = node_dist
        # get the splitting plane
        split_plane = curr_value[split_axis]
        plane_dist2 = (point[split_axis] - split_plane) ** 2

        # Search the side of the splitting plane that the point is in
        if point[split_axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point, k, results, get_dist, counter)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, get_dist, counter)

        # Search the other side of the splitting plane if it may contain
        # points closer than the farthest point in the current results.
        if -plane_dist2 > results[0][0] or len(results) < k:
            if point[split_axis] < curr_value[split_axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, get_dist,
                                            counter)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, get_dist,
                                           counter)

    def search(self, point, k, dist=None):
        if k < 1:
            raise ValueError("k must be greater than 0.")
        if dist is None:
            get_dist = lambda n, p: np.linalg.norm(n - p, ord=2)
        else:
            get_dist = lambda n, p: dist(n.value, p) if isinstance(n, KdNode) else dist(n, p)

        results = []
        self._search_node(point, k, results, get_dist, itertools.count())
        return [(node, p, -d) for d, p, _, node in sorted(results, reverse=True)]


class KdTree:
    def __init__(self, x, leaf_size=30):
        self.x = [i for i in _to_numpy(x)]
        self.leaf_size = leaf_size
        self.tree = self._create_tree(self.x, None)

    @staticmethod
    def _calc_split_axis(x):
        x = np.array(x)
        return sorted([(i, np.var(x[:, i])) for i in range(x.shape[1])],
                      key=lambda v: v[1],
                      reverse=True)[0][0]

    @staticmethod
    def _split_by_axis(x, axis):
        x = sorted(x, key=lambda v: v[axis], reverse=False)
        split_index = len(x) // 2
        return np.array(x[:split_index]), \
               x[split_index], \
               np.array(x[split_index + 1:])

    def _create_tree(self, data, parent):
        if len(data) <= self.leaf_size:
            return KdNode(value=np.array(data))
        split_axis = self._calc_split_axis(data)
        left_data, split_data, right_data = self._split_by_axis(data, split_axis)
        node = KdNode(value=split_data, split_axis=split_axis, parent=parent)
        node.left = self._create_tree(left_data, node) if left_data.shape[0] > 0 else None
        node.right = self._create_tree(right_data, node) if right_data.shape[0] > 0 else None
        return node

    def print_tree(self, tree):
        print(tree.value, tree.split_axis)
        for c in tree.children:
            self.print_tree(c)

    def search(self, point, k=1, dist=None):
        return self.tree.search(point, k, dist)


if __name__ == "__main__":
    x = [[1, 2, 3], [1, 2, 3], [1, 2, 5], [1, 4, 2], [4, 5, 7], [10, 2, 6]]
    kd_tree = KdTree(x, leaf_size=2)
    kd_tree.print_tree(kd_tree.tree)

    print("result")
    for ret in kd_tree.search(point=[1, 2, 3], k=2):
        node = ret[0]
        point = ret[1]
        x = np.array(x)
        print(point)
        index = np.where((x == point).all(1))[0][0]
        print(index)
        # print(node.value, point)
