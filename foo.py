# class A:
#     def __init__(self, a):
#         self.a = a
#
#     def kk(self):
#         return 1
#
#     def foo(self, **params):
#         for i, j in params.items():
#             print(i, j)
#             self.__dict__[i] = j
#         print(self.a)
#
#
# aa = A(1)
#
# d = {"a", 5}
# aa.foo(a=5)
#
# import pandas as pd
# import numpy as np
#
# a = pd.DataFrame({"a": [1, 2]})
# print(np.array(a))
#
# x = [[2, 4.], [2, 4.]]
# b = np.square(np.sqrt(x))
# print(np.array(b).round(12))
# print(b.round(12) == x)
#
# from lemon import ab
# import os, sys
# print(os.getcwd())

from sklearn.datasets import load_breast_cancer, load_boston, load_wine
# import pickle
#
# df = load_wine()
# # print(df)
#
# with open("wine.pkl", "wb") as f:
#     pickle.dump(df, f)
#
# with open("wine.pkl", "rb") as f:
#     df = pickle.load(f)
# print(df.keys())

from sklearn.datasets import load_iris

#
# data = load_iris(True)
# print(data)

# from lemon.datasets import load_iris
#
# print(load_iris())
#
# import pandas as pd
# import numpy as np
#
# df = pd.read_csv("./test/train.csv")
#
# target_names = ["survived", "unsurvived"]
# target = np.array(df["Survived"].tolist())
#
# df = df[df.columns.difference(["Survived"])]
#
# feature_names = df.columns.tolist()
# DESCR = "Titanic datasets from kaggle"
# data = df.to_numpy()
#
# dataset = {
#     "feature_names": feature_names,
#     "DESCR": DESCR,
#     "data": data,
#     "target_names": target_names,
#     "target": target
# }
# import pickle
# print(dataset)
# with open("lemon/datasets/titanic.pkl", "wb") as f:
#     pickle.dump(dataset, f)

from lemon.datasets import load_titanic

x, y = load_titanic(True)
print(x)
print(y)
