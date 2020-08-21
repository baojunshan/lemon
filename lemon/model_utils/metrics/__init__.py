import numpy as np


def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred):
    tp, fn, tn, fp = 0, 0, 0, 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t ==0 and p == 1:
            fp += 1
        else:
            tn += 1
    return tp / (tp + fp)


def recall(y_true, y_pred):
    tp, fn, tn, fp = 0, 0, 0, 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t ==0 and p == 1:
            fp += 1
        else:
            tn += 1
    return tp / (tp + fn)


def f_beta_score(y_true, y_pred, beta=1):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    f_beta = (1 + beta**2) * precision_ * recall_ / (beta ** 2 * precision_ + recall_)
    return f_beta


def f1_score(y_true, y_pred):
    return f_beta_score(y_true, y_pred, 1)


def auc(y_true, y_prob):
    pos = [i for i in range(len(y_true)) if y_true[i] == 1]
    neg = [i for i in range(len(y_true)) if y_true[i] == 0]

    auc = 0
    for i in pos:
        for j in neg:
            if y_prob[i] > y_prob[j]:
                auc += 1
            elif y_prob[i] == y_prob[j]:
                auc += 0.5

    return auc / (len(pos) * len(neg))


def ks(y_true, y_prob):
    sort_value = sorted(zip(y_true, y_prob), key=lambda x: x[1], reverse=True)
    max_ks = 0.0
    total_good = sum(y_true)
    total_bad = len(y_true) - total_good
    good_count, bad_count = 0, 0
    for label in [v[0] for v in sort_value]:
        if label == 0:
            bad_count += 1
        else:
            good_count += 1
        val = abs(bad_count / total_bad - good_count / total_good)
        max_ks = max(max_ks, val)
    return max_ks



