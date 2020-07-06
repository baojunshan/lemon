from lemon.datasets import load_iris
from lemon.model_utils.model_selection import train_test_split
from lemon.model_utils.metrics import accuracy
from lemon.supervised.linear_model import LogisticRegression


x, y = load_iris(x_y=True)
x, y = x[y < 2], y[y < 2]

train_x, test_x, train_y, test_y = train_test_split(x, y)

model = LogisticRegression().fit(train_x, train_y)
pred = model.predict(test_x)
print(accuracy(test_y, pred))
