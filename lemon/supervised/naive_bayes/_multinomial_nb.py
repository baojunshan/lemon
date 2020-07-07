from lemon.base import BaseModel
import numpy as np


class MultinomialNB(BaseModel):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_count = dict()
        self.likelihood_ = dict()
        self.prior_ = dict()

    def fit(self, x, y):
        x = self._to_numpy(x)
        y = self._to_numpy(y)
        self.classes_count = {k: v for k, v in zip(*np.unique(y, return_counts=True))}
        self.prior_ = {k: v / x.shape[0] for k, v in self.classes_count.items()}

        for label in self.classes_count.keys():
            self.likelihood_[label] = dict()
            sub_x = x[y == label]
            for j in range(sub_x.shape[1]):
                cls, cls_count = np.unique(sub_x[:, j], return_counts=True)
                n = cls.shape[0]
                prob = {k: (v + self.alpha) / (self.classes_count[label] + self.alpha * n) \
                           for k, v in zip(cls, cls_count)}
                prob["na"] = 1 / n
                self.likelihood_[label][j] = prob
        return self

    def predict(self, x):
        x = self._to_numpy(x)
        pred = list()
        for curr_x in x:
            posteriors = list()
            for curr_y in self.classes_count.keys():
                likelihood = sum([np.log(self.likelihood_[curr_y][i].get(v, self.likelihood_[curr_y][i]["na"])) \
                                  for i, v in enumerate(curr_x)])
                prior = np.log(self.prior_[curr_y])
                posteriors.append(likelihood + prior)
            pred.append(list(self.classes_count.keys())[np.argmax(posteriors)])
        return pred
