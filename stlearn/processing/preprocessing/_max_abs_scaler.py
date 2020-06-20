from stlearn.base import BasePreprocessor


class MaxAbsScaler(BasePreprocessor):
    def __init__(self):
        self.max_abs_ = None
        self.scale_ = None
        self.n_samples_seen_ = None

    def fit(self, x):
        self._check_type(x)
        x = self._to_numpy(x)
        self.max_abs_ = np.abs(x).max(axis=0)
        self.scale_ = self.max_abs_
        self.n_samples_seen_ = len(x)
        return self

    def transform(self, x):
        x = self._to_numpy(x)
        return x / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        x = self._to_numpy(x)
        return x * self.scale_

    def partial_fit(self, x):
        self._check_type(x)
        x = self._to_numpy(x)
        self.max_abs_ = np.maximum(self.max_abs_, np.abs(x).max(axis=0))
        self.scale_ = self.max_abs_
        self.n_samples_seen_ += len(x)
        return self


if __name__ == "__main__":
    import numpy as np
    from sklearn.preprocessing import MaxAbsScaler as MA
    X = [[1., -1., 2.],
         [2., 0., 0.],
         [0., 1., -4.],
         [0., 1., -4.]]

    transformer = MA().fit(X)
    print(transformer.transform(X))
    print(transformer.max_abs_)
    print(transformer.scale_)
    print(transformer.n_samples_seen_)

    my_transformer = MaxAbsScaler().fit(X)
    print(my_transformer.transform(X))
    print(my_transformer.max_abs_)
    print(my_transformer.scale_)
    print(my_transformer.n_samples_seen_)
