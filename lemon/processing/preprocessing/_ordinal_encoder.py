from lemon.base import BasePreprocessor
import numpy as np


class OrdinalEncoder(BasePreprocessor):
    def __init__(self):
        self.categories_ = None

    def fit(self, x):
        self._check_type(x)
        x = self._to_numpy(x)
        self.categories_ = [np.unique(v, axis=0) for v in x.T]
        return self

    def transform(self, x):
        x = self._to_numpy(x)
        ret = [[list(self.categories_[i]).index(v) for v in col] for i, col in enumerate(x.T)]
        return np.array(ret).T

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        x = self._to_numpy(x)
        ret = [[self.categories_[i][int(j)] for j in col] for i, col in enumerate(x.T)]
        return np.array(ret).T


if __name__ == "__main__":
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    encoder = OrdinalEncoder()
    encoder.fit(X)
    print(encoder.categories_)
    print(encoder.transform([['Female', 3], ['Male', 1]]))
    print(encoder.inverse_transform([[0, 2], [1, 0]]))
