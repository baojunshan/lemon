from lemon.datasets import load_iris
from lemon.model_utils.model_selection import train_test_split
from lemon.supervised.naive_bayes import GaussianNB
from lemon.model_utils.metrics import accuracy


x, y = load_iris(x_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, split_rate=0.8, random_state=2020)

model = GaussianNB()
model.fit(x_train, y_train)
pred = model.predict(x_test)

print(accuracy(y_test, pred))

