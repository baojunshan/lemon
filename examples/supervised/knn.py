from lemon.datasets import load_iris
from lemon.model_utils.model_selection import train_test_split
from lemon.model_utils.metrics import accuracy
from lemon.supervised.neighbors import KNearestNeighbors

x, y = load_iris(x_y=True)
train_x, test_x, train_y, test_y = train_test_split(x, y)

model1 = KNearestNeighbors(n_neighbors=5).fit(train_x, train_y)
pred = model1.predict(test_x)
print(accuracy(test_y, pred))

model2 = KNearestNeighbors(n_neighbors=5, algorithm="kd_tree", leaf_size=5).fit(train_x, train_y)
pred = model2.predict(test_x)
print(accuracy(test_y, pred))
