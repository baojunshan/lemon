from stlearn.base import BasePreprocessor


class Binarizer(BasePreprocessor):
    def __init__(self, threshold=0):
        self.threshold = threshold

    def fit(self, x):
        self._check_trans_input(x)
        return self

    def transform(self, x):
        x = self._check_trans_input(x)
        x = (x > self.threshold).astype('float')
        return x

    def fit_transform(self, x):
        return self.transform(x)