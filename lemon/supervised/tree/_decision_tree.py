from lemon.base import BaseModel

import numpy as np
from collections import namedtuple
from abc import ABCMeta, abstractmethod, abstractstaticmethod

# 决策树结点
# Parameters
# ----------
# feature : 特征，需要进行比对的特征名
# val : 特征值，当特征为离散值时，如果对应的特征值等于val，将其放入左子树，否则放入右子树
# left : 左子树
# right : 右子树
# label : 所属的类
TreeNode = namedtuple("TreeNode", 'feature val left right label')


class DecisionTree(BaseModel):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def build(self, X_, features, depth=None):

        if self.unique_cond(X_):
            return TreeNode(None, None, None, None, self.get_target(X_))
        if features.shape[0] == 0 or depth and depth >= self.max_depth:
            return TreeNode(None, None, None, None, self.stop_early_target(X_))
        feature, val = self.get_best_index(X_, features)
        new_features = features[features != feature]
        del features
        left, right = self.devide(X_, feature, val)
        if left.any():
            left_branch = self.build(left, new_features, depth + 1 if depth else None)
        else:
            left_branch = TreeNode(None, None, None, None, val)
        if right.any():
            right_branch = self.build(right, new_features, depth + 1 if depth else None)
        else:
            right_branch = TreeNode(None, None, None, None, val)
        return TreeNode(feature, val, left_branch, right_branch, None)

    def fit(self, X, y):
        features = np.arange(X.shape[1])
        X_ = np.c_[X, y]
        self.root = self.build(X_, features)
        return self

    def predict_one(self, x):
        p = self.root
        while p.label is None:
            # print('feature', x[p.feature])
            # print(p.val)
            p = p.left if self.judge(x[p.feature], p.val) else p.right
        return p.label

    def predict(self, X):
        """
        :param X: shape = [n_samples, n_features]
        :return: shape = [n_samples]
        """
        return np.array([self.predict_one(x) for x in X])

    @abstractstaticmethod
    def devide(X_, feature, val):
        pass

    @abstractmethod
    def unique_cond(self, X_):
        pass

    @abstractmethod
    def get_target(self, X_):
        pass

    @abstractmethod
    def stop_early_target(self, X_):
        pass

    @abstractmethod
    def judge(self, node_val, val):
        pass

    @abstractmethod
    def get_best_index(self, X_, features):
        pass


class DecisionTreeClassifier(DecisionTree):
    def get_best_index(self, X_, features):
        ginis = [DecisionTreeClassifier.get_fea_best_val(
            np.c_[X_[:, i], X_[:, -1]]) for i in features]
        ginis = np.array(ginis)
        i = np.argmax(ginis[:, 1])
        return features[i], ginis[i, 0]

    def unique_cond(self, X_):
        return True if np.unique(X_[:, -1]).shape[0] == 1 else False

    def judge(self, node_val, val):
        return True if node_val == val else False

    def stop_early_target(self, X_):
        classes, classes_count = np.unique(X_[:, -1], return_counts=True)
        return classes[np.argmax(classes_count)]

    def get_target(self, X_):
        return X_[0, -1]

    @staticmethod
    def devide(X_, feature, val):
        return X_[X_[:, feature] == val], X_[X_[:, feature] != val]

    @staticmethod
    def gini(D):
        """求基尼指数 Gini(D)
        :param D: shape = [ni_samples]
        :return: Gini(D)
        """
        # 目前版本的 numpy.unique 不支持 axis 参数
        _, cls_counts = np.unique(D, return_counts=True)
        probability = cls_counts / cls_counts.sum()
        return 1 - (probability ** 2).sum()

    @staticmethod
    def congini(D_, val):
        """求基尼指数 Gini(D, A)
        :param D_: 被计算的列. shape=[ni_samples, 2]
        :param val: 被计算的列对应的切分变量
        :return: Gini(D, A)
        """
        left, right = D_[D_[:, 0] == val], D_[D_[:, 0] != val]
        return DecisionTreeClassifier.gini(left[:, -1]) * left.shape[0] / D_.shape[0] + \
               DecisionTreeClassifier.gini(right[:, -1]) * right.shape[0] / D_.shape[0]

    @staticmethod
    def get_fea_best_val(D_):
        """寻找当前特征对应的最优切分变量
        :param D_: 被计算的列. shape=[ni_samples, 2]
        :return: 最优切分变量的值和基尼指数的最大值
        """
        vals = np.unique(D_[:, :-1])
        tmp = np.array([DecisionTreeClassifier.congini(D_, val) for val in vals])
        return vals[np.argmax(tmp)], tmp.max()


class DecisionTreeRegressor(DecisionTree):
    def get_best_index(self, X_, features):
        losses = np.array([DecisionTreeRegressor.feature_min_loss(X_, feature) for feature in features])
        i = np.argmin(losses[:, 1])
        return features[i], losses[i, 0]

    def unique_cond(self, X_):
        return True if X_[:, -1].std() <= 0.1 else False

    def judge(self, node_val, val):
        return True if node_val <= val else False

    def stop_early_target(self, X_):
        self.get_target(X_)

    def get_target(self, X_):
        return X_[:, -1].mean()

    @staticmethod
    def devide(X_, feature, val):
        return X_[X_[:, feature] <= val], X_[X_[:, feature] >= val]

    @staticmethod
    def feature_loss(X_, feature, val):
        left, right = DecisionTreeRegressor.devide(X_, feature, val)
        left_loss = np.sum((left[:, -1] - left[:, -1].mean()) ** 2)
        right_loss = np.sum((right[:, -1] - right[:, -1].mean()) ** 2)
        return left_loss + right_loss

    @staticmethod
    def feature_min_loss(X_, feature):
        losses = np.array(list(map(lambda val: DecisionTreeRegressor.feature_loss(X_, feature, val), X_[:, feature])))
        i = np.argmin(losses)
        return X_[i, feature], losses[i]


# class TreeNode:
#     def __init__(self, feature, value, left, right):
#         self.feature = feature
#         self.children = list()
#         self.value = value
#         self.left = left
#         self.right = right


# class DecisionTree(BaseModel):
#     def __init__(self, criterion='gini', max_depth=None, min_sample_split=2):
#         self.criterion = criterion
#         self.max_depth = None
#         self.min_sample_split = min_sample_split
#
#     @staticmethod
#     def _entropy(x):
#         classes, class_count = np.unique(x, return_counts=True)
#         return -1 * sum([c / len(x) * np.log(c / len(x)) for c in class_count])
#
#     @classmethod
#     def _cond_entropy(cls, x, y):
#         l = len(y)
#         classes, class_count = np.unique(x, return_counts=True)
#         ret = 0
#         for c, n in zip(classes, class_count):
#             ret += cls._entropy(y[np.where(x == c)]) * n / l
#         return ret
#
#     @staticmethod
#     def _gini(x):
#         classes, class_count = np.unique(x, return_counts=True)
#         l = len(x)
#         return 1 - sum([(c / l) ** 2 for c in class_count])
#
#     @classmethod
#     def _cond_gini(cls, x, y):
#         l = len(y)
#         classes, class_count = np.unique(x, return_counts=True)
#         ret = 0
#         for c, n in zip(classes, class_count):
#             ret += cls._gini(y[np.where(x == c)]) * n / l
#         return ret
#
#     @classmethod
#     def _info_gain(cls, x, y):
#         return cls._entropy(y) - cls._cond_entropy(x, y)
#
#     @classmethod
#     def _gain_ratio(cls, x, y):
#         return (cls._entropy(y) - cls._cond_entropy(x, y)) / cls._entropy(y)
#
#     @classmethod
#     def _inverse_gini_index(cls, x, y):
#         # gini smaller is better -> inverse bigger is better
#         return -1 * cls._cond_gini(x, y)
#
#     def fit(self, x, y):
#         max_feat = -1
#         max_gain = float("-inf")
#         for i in range(x.shape[1]):
#             x_ = x[:, i]
#             if self.criterion == "info_gain":
#                 curr_gain = self._info_gain(x_, y)
#             elif self.criterion == "gain_ratio":
#                 curr_gain = self._gain_ratio(x_, y)
#             elif self.criterion == "gini":
#                 curr_gain = self._inverse_gini_index(x_, y)
#             else:
#                 raise ValueError("criterion should be correct!")
#             if curr_gain > max_gain:
#                 max_gain = curr_gain
#                 max_feat = i
#
#     def predict(self, x):
#         pass


if __name__ == "__main__":
    x = [['青年', '否', '否', '一般', '否'],
         ['青年', '否', '否', '好', '否'],
         ['青年', '是', '否', '好', '是'],
         ['青年', '是', '是', '一般', '是'],
         ['青年', '否', '否', '一般', '否'],
         ['中年', '否', '否', '一般', '否'],
         ['中年', '否', '否', '好', '否'],
         ['中年', '是', '是', '好', '是'],
         ['中年', '否', '是', '非常好', '是'],
         ['中年', '否', '是', '非常好', '是'],
         ['老年', '否', '是', '非常好', '是'],
         ['老年', '否', '是', '好', '是'],
         ['老年', '是', '否', '好', '是'],
         ['老年', '是', '否', '非常好', '是'],
         ['老年', '否', '否', '一般', '否'],
         ]
    columns = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    y = [i[-1] for i in x]
    x = [i[:-1] for i in x]

    model = DecisionTree()
