from ...base import BaseImputer

import numpy as np


class SimpleImputer(BaseImputer):
    def __init__(self, missing_values=np.nan, strategy='mean'):
        self.missing_value = {np.nan}
        if not isinstance(missing_values, list):
            self.missing_value.add(missing_values)
        else:
            self.missing_value |= set(missing_values)
        self.missing_value.remove(np.nan)
        self.strategy = strategy
        self.cache = list()

    def fit(self, x):
        if self._check_type(x):
            x = self._to_numpy(x)

        filter_ = "|".join(["(col=={})".format(m) for m in self.missing_value])
        for col in x.T:
            index = np.where(eval(filter_))
            if self.strategy == "mean":
                value = np.mean(col[~index])
            elif self.strategy == "max":
                value = np.max(col[~index])
            elif self.strategy == "min":
                value = np.min(col[~index])
            else:
                value = 0
            self.cache.append(value)
        return self

    def transform(self, x):
        if self._check_type(x):
            x = self._to_numpy(x)
        ret = list()
        filter_ = "|".join(["(col=={})".format(m) for m in self.missing_value])
        for col, value in zip(x.T, self.cache):
            index = np.where(eval(filter_))
            col[index] = value
            col[np.isnan(col)] = value
            ret.append(col)
        return np.array(ret).T

    def fit_transform(self, x):
        return self.fit(x).transform(x)
