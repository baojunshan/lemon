import math


def minkowski_distance(x1, x2, p=2):
    if len(x1) != len(x2):
        return -1

    if isinstance(p, int):
        sum_ = 0
        for i in range(len(x1)):
            sum_ += abs(x1[i] - x2[i]) ** p
        return sum_ ** (1 / p)
    elif p == "inf":
        return chebyshev_distance(x1, x2)
    return -1


def chebyshev_distance(x1, x2):
    if len(x1) == len(x2):
        max_ = 0
        for i in range(len(x1)):
            d = abs(x1[i] - x2[i])
            max_ = d if max_ < d else max_
        return max_
    return -1


def manhattan_distance(x1, x2):
    return minkowski_distance(x1, x2, p=1)


def euclidean_distance(x1, x2):
    return minkowski_distance(x1, x2, p=2)


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 \
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r * 1000
