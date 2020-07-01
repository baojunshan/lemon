from lemon.supervised.linear_model import Perceptron
from lemon.datasets import load_breast_cancer
import numpy as np

x, y = load_breast_cancer(x_y=True)
y = np.where(y > 0, 1, -1)

train_x, test_x = x[:400], x[400:]
train_y, test_y = y[:400], y[400:]

print(test_x)

model = Perceptron(verbose=True)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
print(model.score(test_y, pred_y))


from sklearn.linear_model.perceptron import Perceptron as P
m = P(verbose=True)
m.fit(train_x, train_y)
p = model.predict(test_x)
print(model.score(test_y, p))

