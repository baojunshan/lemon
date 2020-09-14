# lemon 🍋🍋🍋
一个轻量级的机器学习框架（纯python+numpy实现的迷你版scikit-learn）

# examples
```python
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
```

# 当前进度
- [x] datasets(boston, breast_canner, iris, titanic, wine)
- [x] processing
    - [x] preprocessing(binarizer, transformer, discretizer, encoding, scaler)
    - [x] decomposition(pca, svd)
    - [x] impute(simple-impute)
- [x] feature_utils
    - [x] feature_selection (woe_iv)
- [x] model_utils
    - [x] metrics
    - [x] model_selection(psi)
- [ ] supervised
    - [x] linear_model(simple-linear, lasso, ridge, elastic-net, perceptron, logistic-regression)
    - [x] naive_bayes(gaussian, multinomial)
    - [x] neighbors(kd-tree-based-knn)
    - [x] svm
    - [x] tree
    - [ ] hmm
    - [ ] crf
    - [ ] ensemble
- [x] semi_supervised
    - [x] label_propagation
    - [x] louvain
    - [x] pagerank
- [x] unsupervised
    - [x] dbscan
    - [x] kmeans(simple-kmeans, kmeans++, mini-batch-kmeans)
    
    


