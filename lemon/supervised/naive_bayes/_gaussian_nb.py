from lemon.base import BaseModel
import numpy as np


class GaussianNB(BaseModel):
    def __init__(self):
        self.classes = None
        self.classes_count = None
        self.prior_ = None
        self.mean = None
        self.var = None

    @staticmethod
    def gauss_func(x, mean, std):
        var = std @ std
        numerator = -np.exp(np.sum((x - mean) ** 2, axis=1) / (2 * var))
        return numerator / np.sqrt(2 * np.pi * var)

    def fit(self, x, y):
        self.classes, self.classes_count = np.unique(y, return_counts=True)
        self.prior_ = {k: v / x.shape[0] for k, v in zip(self.classes, self.classes_count)}
        self.mean = np.zeros((self.classes.shape[0], x.shape[1]), dtype=np.float64)
        self.var = np.zeros((self.classes.shape[0], x.shape[1]), dtype=np.float64)
        for i, label in enumerate(self.classes):
            x_i = x[y == label]
            self.mean[i, :] = np.mean(x_i, axis=0)
            self.var[i, :] = np.var(x_i, axis=0)
        return self

    def predict(self, x):
        x = self._to_numpy(x)
        likelihood = []
        for i in range(self.classes.shape[0]):
            likelihood.append(self.classes_count[i] * self.gauss_func(x, self.mean[i, :], self.var[i, :]))
        likelihood = np.array(likelihood).T
        return self.classes[np.argmax(likelihood, axis=1)]


if __name__ == "__main__":
    x = np.random.random((3, 5))

    y = [1, 1, 0]
    nb = GaussianNB().fit(x, y).predict(x)
    print(nb)
