from stlearn.base import BasePreprocessor
import numpy as np


class BeforeFitError(Exception):
    pass


class OneHotEncoder(BasePreprocessor):
    def __init__(self, drop=None, handle_unknown="error"):
        self.categories_ = None
        self.drops_ = None
        self.drop = drop
        self.handle_unknow = handle_unknown

    def _update_categories(self, x):
        self.categories_ = list()
        for i in range(len(x[0])):
            self.categories_.append(list(set(
                line[i] for line in x
            )))
        if self.drop is None:
            self.drops_ = [None] * len(self.categories_)
            return
        if self.drop == "first":
            self.drops_ = [c[0] for c in self.categories_]
            self.categories_ = [c[1:] for c in self.categories_]
            return
        if self.drop == "if_binary":
            self.drops_ = [c[0] if len(c) == 2 else None for c in self.categories_]
            self.categories_ = [c[1:] if len(c) == 2 else c for c in self.categories_]
            return

    def fit(self, x):
        self._check_type(x)
        self._update_categories(x)
        return self

    def transform(self, x):
        x = self._to_numpy(x)
        ret = list()
        for line in x:
            line_onehot = list()
            for i in range(len(line)):
                if self.handle_unknow == "error":
                    if line[i] not in self.categories_[i] and line[i] != self.drops_[i]:
                        raise ValueError("{} not in columns!".format(line[i]))
                temp = [0] * len(self.categories_[i])
                if line[i] in self.categories_[i]:
                    temp[self.categories_[i].index(line[i])] = 1
                line_onehot += temp
            ret.append(line_onehot)
        return np.array(ret)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        ret = list()
        interval = [0]
        for c in self.categories_:
            interval.append(interval[-1] + len(c))
        for line in x:
            line_ret = list()
            for i in range(len(self.categories_)):
                inter = line[interval[i]: interval[i+1]]
                if 1 in inter:
                    line_ret.append(self.categories_[i][inter.index(1)])
                else:
                    line_ret.append(None)
            ret.append(line_ret)
        return np.array(ret)

    def get_feature_names(self, input_features):
        if self.categories_ is None:
            raise BeforeFitError("You should fit first!")
        if len(input_features) != len(self.categories_):
            raise ValueError("The length of input feature should be the same with x")
        ret = list()
        for i in range(len(input_features)):
            ret += [str(input_features[i]) + "_" + str(c) for c in self.categories_[i]]
        return np.array(ret)


if __name__ == "__main__":
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)
    print(enc.transform([['Female', 1], ['Male', 4]]))
    print(enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]))
    print(enc.get_feature_names(["gender", "group"]))

    from sklearn.preprocessing import OneHotEncoder as OE
    oe = OE(handle_unknown='ignore')
    oe.fit(X)
    print(oe.transform([['Female', 1], ['Male', 4]]).toarray())
    print(oe.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]))
    print(oe.get_feature_names(["gender", "group"]))


