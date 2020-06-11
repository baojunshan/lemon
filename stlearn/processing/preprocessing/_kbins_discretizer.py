from stlearn.base import BasePreprocessor


class KBinsDiscretizer(BasePreprocessor):
    def __init__(self, n_bins=5, encode="onehot", strategy="quantile"):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

    def fit(self, x):
        pass

    def transform(self, x):
        pass

    def fit_transform(self, x):
        pass

    def inv(self):
        pass
