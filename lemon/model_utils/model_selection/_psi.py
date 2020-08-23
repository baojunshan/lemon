import numpy as np


def psi(old, new, bins=10):
    len_o, len_n = len(old), len(new)
    bin_len = (old.max() - old.min()) / bins
    cuts = [old.min() + i * bin_len for i in range(1, bins)]
    cuts.insert(0, -float("inf"))
    cuts.append(float("inf"))
    old_cuts = np.histogram(old, bins=cuts)
    new_cuts = np.histogram(new, bins=cuts)

    psi = 0
    for o, n in zip(old_cuts, new_cuts):
        psi += (len(o) / len_o - len(n) / len_n) * np.log(len(o) / len_o / (len(n) / len_n))
    return psi
