from lemon.base import BasePreprocessor

import numpy as np


class MinMaxScaler(BasePreprocessor):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_samples_seen_ = None

    def fit(self, x):
        self._check_type(x)
        x = self._to_numpy(x)
        self.data_min_ = np.abs(x).min(axis=0)
        self.data_max_ = np.abs(x).max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.feature_range[0] - self.feature_range[0]) / self.data_range_
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self.n_samples_seen_ = len(x)
        return self

    def transform(self, x):
        x = self._to_numpy(x)
        return (x - self.data_min_) / self.data_range_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        x = self._to_numpy(x)
        return (x - self.data_min_) * self.scale_

    def partial_fit(self, x):
        self._check_type(x)
        x = self._to_numpy(x)
        self.data_min_ = np.minimum(self.data_min_, np.abs(x).min(axis=0))
        self.data_max_ = np.maximum(self.data_max_, np.abs(x).max(axis=0))
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.feature_range[0] - self.feature_range[0]) / self.data_range_
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self.n_samples_seen_ += len(x)
        return self


if __name__ == "__main__":
    from sklearn.preprocessing import MinMaxScaler as MMS

    X = [[-1, 2],
         [-0.5, 6],
         [0, 10],
         [1, 18]]
    scaler = MMS()
    print(scaler.fit(X).transform(X))
    print(scaler.scale_)
    print(scaler.min_)

    my_scaler = MinMaxScaler()
    print(scaler.fit(X).transform(X))
    print(scaler.scale_)
    print(scaler.min_)
