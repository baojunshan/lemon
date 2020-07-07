from lemon.datasets import load_iris
from lemon.model_utils.model_selection import train_test_split
from lemon.model_utils.metrics import accuracy
from lemon.supervised.naive_bayes import GaussianNB, MultinomialNB

import numpy as np
x, y = load_iris(x_y=True)
x = np.round(x)

train_x, test_x, train_y, test_y = train_test_split(x, y)

mnb = MultinomialNB().fit(train_x, train_y)
pred = mnb.predict(train_x)
print(accuracy(train_y, pred))

gnb = GaussianNB().fit(train_x, train_y)
pred = gnb.predict(train_x)
print(accuracy(train_y, pred))


