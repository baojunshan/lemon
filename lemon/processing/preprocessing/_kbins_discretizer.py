import numpy as np
from lemon.base import BasePreprocessor
from lemon.processing.preprocessing._onehot_encoder import OneHotEncoder

# from lemon.cluster import KMeans


class KBinsDiscretizer(BasePreprocessor):
    def __init__(self, n_bins=5, encode="onehot", strategy="quantile"):
        if n_bins < 2:
            raise ValueError("b_bins should be larger then 2!")
        if encode not in ('onehot', 'ordinal'):
            raise ValueError("encode should be 'onehot' or 'ordinal'!")
        if strategy not in ('quantile', 'uniform', 'kmeans'):
            raise ValueError("strategy should be 'quantile' or 'uniform' or 'kmeans'!")
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

        self.n_bins_ = None
        self.bin_edges_ = None

    def fit(self, x):
        x = np.array(x)
        n_features = x.shape[1]
        self.n_bins_ = [self.n_bins] * len(x[0])
        self.bin_edges_ = [list()] * len(x[0])
        for i in range(n_features):
            col = x[:, i]
            if self.strategy == "quantile":
                quantiles = np.linspace(0, 100, self.n_bins_[i] + 1)
                self.bin_edges_[i] = np.asarray(np.percentile(col, quantiles))
            elif self.strategy == "uniform":
                max_ = max(col)
                min_ = min(col)
                self.bin_edges_[i] = np.linspace(min_, max_, self.n_bins_[i] + 1)
            elif self.strategy == "kmeans":
                pass
        return self

    def transform(self, x):
        x = np.array(x)
        n_feature = x.shape[1]
        ret = list()
        for i in range(n_feature):
            col = x[:, i]
            edges = self.bin_edges_[i]
            real_col = list()
            for v in col:
                num = self.n_bins
                for index in range(len(edges)):
                    if edges[index] > v:
                        num = index
                        break
                real_col.append(num - 1)
            ret.append(real_col)
        ret_array = np.array(ret).T
        if self.encode == "onehot":
            return OneHotEncoder(drop=None, handle_unknown="ignore").fit_transform(ret_array)
        return ret_array

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


if __name__ == "__main__":
    from sklearn.preprocessing import KBinsDiscretizer as kb

    X = [[-2, 1, -4, -1],
         [-1, 2, -3, -0.5],
         [0, 3, -2, 0.5],
         [1, 4, -1, 2]]
    est = kb(n_bins=2, encode='onehot-dense', strategy='quantile')
    est.fit(X)
    print(est.transform(X))
    print(est.n_bins_)
    print(est.bin_edges_)

    kbins = KBinsDiscretizer(n_bins=2, encode='onehot', strategy='quantile')
    kbins.fit(X)
    print(kbins.transform(X))
    print(kbins.n_bins_)
    print(kbins.bin_edges_)
