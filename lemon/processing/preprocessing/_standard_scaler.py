from lemon.base import BasePreprocessor
import numpy as np


class StandardScaler(BasePreprocessor):
    def __init__(self):
        self.scale_ = None
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = None

    def fit(self, x):
        self._check_type(x)
        x = self._to_numpy(x)
        self.mean_ = np.mean(x, axis=0)
        self.var_ = np.std(x, axis=0) ** 2
        self.scale_ = np.sqrt(self.var_)
        self.n_samples_seen_ = len(x)
        return self

    def transform(self, x):
        x = self._to_numpy(x)
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_

    def partial_fit(self, x):
        pass


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler as SS

    X = [[3, 0], [3, 0], [1, 1], [1, 1]]
    scaler = SS()
    print(scaler.fit(X).transform(X))

    my_scaler = StandardScaler()
    print(my_scaler.fit(X).transform(X))
