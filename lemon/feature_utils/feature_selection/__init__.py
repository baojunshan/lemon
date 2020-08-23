import numpy as np


def simple_woe_iv(bins, label):
    good = sum(label)
    bad = len(label) - good
    start = 0
    ret = 0
    for bin in bins:
        end = start + bin
        sub_good = sum(label[start: end])
        sub_bad = len(label[start: end]) - sub_good
        ret += ((sub_good/good) - (sub_bad/bad)) * np.log((sub_good/good)/(sub_bad/bad))
    return ret


def variance_threshold(x, threshold=0.16):
    ret_cols = list()
    for i, col in enumerate(x.T):
        if np.var(col) > threshold:
            ret_cols.append(i)
    return x.T[ret_cols].T

