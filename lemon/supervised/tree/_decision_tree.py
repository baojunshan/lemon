from lemon.base import BaseModel

import numpy as np


class DecisionTree(BaseModel):
    def __init__(self, criterion='gini', max_depth=None, min_sample_split=2):
        pass

    @staticmethod
    def _entropy(x):
        classes, class_count = np.unique(x, return_counts=True)
        return -1 * sum([c/len(x) * np.log(c/len(x)) for c in class_count])

    def _info_gain(self):
        pass

    def _gain_ratio(self):
        pass

    def _gini(self):
        pass

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


if __name__ == "__main__":
    x = [['青年', '否', '否', '一般', '否'],
        ['青年', '否', '否', '好', '否'],
        ['青年', '是', '否', '好', '是'],
        ['青年', '是', '是', '一般', '是'],
        ['青年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '好', '否'],
        ['中年', '是', '是', '好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '好', '是'],
        ['老年', '是', '否', '好', '是'],
        ['老年', '是', '否', '非常好', '是'],
        ['老年', '否', '否', '一般', '否'],
        ]
    columns = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    y = [i[-1] for i in x]
    x = [i[:-1] for i in x]

    model = DecisionTree()
    print(model._entropy(y))