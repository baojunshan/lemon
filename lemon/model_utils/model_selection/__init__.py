import pandas as pd
import numpy as np
import random


def _to_numpy(x):
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array(x)
    raise TypeError("{}'s input should be pandas dataframe or numpy array!".format(self.__class__.__name__))


def train_test_split(*arrays, split_rate=0.75, random_state=2020, shuffle=True):
    data = [_to_numpy(d) for d in arrays]
    if shuffle:
        for d in data:
            random.seed(random_state)
            random.shuffle(d)
    ret = []
    for d in data:
        ret.append(d[:int(d.shape[0] * split_rate)])
        ret.append(d[int(d.shape[0] * split_rate):])
    return ret
