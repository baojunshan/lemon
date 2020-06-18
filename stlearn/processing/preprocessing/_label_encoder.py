from stlearn.base import BasePreprocessor
import numpy as np


class LabelEncoder(BasePreprocessor):
    def __init__(self):
        self.classes_ = None

    def fit(self, x):
        self._check_type(x)
        self.classes_ = np.unique(list(x))
        return self

    def transform(self, x):
        x = self._to_numpy(x)
        return np.array([list(self.classes_).index(v) for v in x])

    def fit_transform(self, x):
        return self.fit(x).transform(x)


if __name__ == "__main__":
    le = LabelEncoder()
    le.fit(["a", "b", "c"])

    print(le.classes_)
    print(le.transform(["a", "b", "c"]))
