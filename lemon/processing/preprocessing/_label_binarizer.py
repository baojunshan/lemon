from lemon.base import BasePreprocessor
import numpy as np


class LabelBinarizer(BasePreprocessor):
    def __init__(self, pos_label=1, neg_label=0):
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.classes_ = None

    def fit(self, x):
        self._check_type(x)
        self.classes_ = np.unique(list(x))
        return self

    def transform(self, x):
        self.fit(x)
        x = self._to_numpy(x)
        if len(self.classes_) < 3:
            check_val = self.classes_[0]
            ret = [self.neg_label if check_val == i else self.pos_label for i in x]
        else:
            ret = np.ones([len(x), len(self.classes_)]) * self.neg_label
            for i in range(len(x)):
                ret[i][list(self.classes_).index(x[i])] = self.pos_label
        return ret

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        self._check_type(x)
        x = self._to_numpy(x)
        if len(self.classes_) == 1:
            return np.array([self.classes_[0]] * len(x))
        if len(self.classes_) == 2:
            return np.array([self.classes_[1] if x[i][0] == self.pos_label else self.classes_[0] for i in range(len(x))])
        else:
            return np.array([self.classes_[list(x[i]).index(self.pos_label)] for i in range(len(x))])


if __name__ == "__main__":
    X = [1, 2, 3]
    from sklearn.preprocessing import LabelBinarizer as LB

    lb = LB(pos_label=3, neg_label=2)
    lb.fit(X)
    print(lb.transform(X))
    print(lb.inverse_transform(np.array([[3], [3], [3]])))
    print(lb.classes_)
    print("-"*10)

    mlb = LabelBinarizer(pos_label=3, neg_label=2)
    print(mlb.fit_transform(X))
    print(mlb.inverse_transform(np.array([[3], [3], [3]])))
    print(mlb.classes_)
