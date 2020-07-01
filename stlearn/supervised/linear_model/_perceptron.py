from stlearn.base import BaseClassifierModel
import numpy as np
import multiprocessing as mp


class Perceptron(BaseClassifierModel):
    def __init__(self,
                 max_iter=1000,
                 tol=0,
                 n_iter_no_change=5,
                 early_stopping=False,
                 n_jobs=None,
                 lr=1,
                 verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping
        self.n_jobs = n_jobs
        self.lr = lr
        self.verbose = verbose

        self.weights_ = None
        self.bias_ = None

    @property
    def _cpu_num(self):
        if self.n_jobs is None or self.n_jobs > 0:
            return self.n_jobs
        if self.n_jobs == -1:
            return None
        raise ValueError("\"n_jobs\" should be '-1', 'None', or number larger than 0!")

    def _func(self, x):
        return self.weights_ @ x.T + self.bias_

    def _loss(self, x, y):
        return self._func(x) * y

    def fit(self, x, y):
        self.weights_ = np.zeros(len(x[0]), dtype="float")
        self.bias_ = 0
        x = self._to_numpy(x)
        y = self._to_numpy(y)

        best_iter = -1
        best_weights = self.weights_
        best_bias = self.bias_
        min_error = x.shape[0]
        no_change_num = 0

        for i in range(self.max_iter):
            if self.n_jobs is None:
                loss = self._loss(x, y)
            else:
                with mp.Pool(self._cpu_num) as p:
                    loss = p.starmap(self._loss, zip(x, y))
            error_count = sum(np.array(loss) <= 0)

            if self.verbose:
                print("iter: {iter}, error_count: {err}, weights: {w}, bias: {b}" \
                      .format(iter=i, err=error_count, w=self.weights_, b=self.bias_))

            if error_count < min_error - self.tol:
                best_iter = i
                best_weights = self.weights_
                best_bias = self.bias_
                min_error = error_count
                no_change_num = 0
            else:
                no_change_num += 1

            # check if need stop:
            if error_count == 0 or \
                    (self.early_stopping and no_change_num >= self.n_iter_no_change):
                print("Best iter: {iter}, error_count: {err}, weights: {w}, bias: {b}" \
                      .format(iter=best_iter, err=min_error, w=best_weights, b=best_bias))
                break
            # update weights and bias
            update_x = x[np.array(loss) <= 0][0]
            update_y = y[np.array(loss) <= 0][0]
            self.weights_ += self.lr * np.dot(update_x, update_y)
            self.bias_ += self.lr * update_y

        # update best parameters
        self.weights_ = best_weights
        self.bias_ = best_bias
        return self

    def predict(self, x):
        x = self._to_numpy(x)
        if self.n_jobs is None:
            ret = np.dot(self.weights_, x.T) + self.bias_
        else:
            with mp.Pool(self._cpu_num) as p:
                ret = p.map(self._func, x)
                ret = np.array(ret)
        return np.where(ret > 0, 1, -1)


if __name__ == "__main__":
    x = [[3, 3], [4, 3], [1, 1]]
    y = [1, 1, -1]
    model = Perceptron(verbose=True)
    model.fit(x, y)
    print(model.predict([[3, 3], [1, 1]]))
