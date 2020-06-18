from stlearn.base import BasePreprocessor


class Binarizer(BasePreprocessor):
    def __init__(self, threshold=0):
        self.threshold = threshold

    def fit(self, x):
        self._check_type(x)
        return self

    def transform(self, x):
        x = self._to_numpy(x)
        x = (x > self.threshold).astype('float')
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
