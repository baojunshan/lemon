from lemon.base import BasePreprocessor
from functools import reduce


class PloynomialFeatures(BasePreprocessor):
    def __init__(self, interaction_only=False, degree=2):
        self.interaction_only = interaction_only
        self.degree = degree
        self.n_input_features_ = None
        self.n_output_features_ = None
        self.powers_ = None

    @staticmethod
    def interaction(a, b):
        ret = []
        for i in a:
            for j in b:
                ret.append(i + j)
        return np.flip(np.unique(ret, axis=0), axis=0)

    def _get_powers(self):
        if self.n_input_features_ is None:
            raise ValueError("Please fit first!")
        inputs = np.zeros([self.n_input_features_, self.n_input_features_])
        for i in range(self.n_input_features_):
            inputs[i][i] = 1

        concat_list = [np.zeros(self.n_input_features_)]
        for d in range(self.degree):
            powers = inputs.copy()
            for i in range(d):
                powers = self.interaction(powers, inputs)
            concat_list.append(powers)
        ret = np.vstack(concat_list).astype('int')
        if self.interaction_only:
            return ret[sum((ret > 1).T) == 0]
        return ret

    @staticmethod
    def _dot_power(ftr, power):
        temp = []
        for p, f in zip(power, ftr):
            temp.append(np.power(f, p))
        return reduce(lambda x, y: x * y, temp)

    def fit(self, x):
        self._check_type(x)
        x = self._to_numpy(x)
        self.n_input_features_ = len(x[0])
        self.powers_ = self._get_powers()
        self.n_output_features_ = self.powers_.shape[0]
        return self

    def transform(self, x):
        self._check_type(x)
        x = self._to_numpy(x)
        if len(x[0]) != self.n_input_features_:
            raise ValueError("Input feature is not equal to fit feature!")

        ret = list()
        for power in self.powers_:
            ret.append(self._dot_power(x.T, power))
        return np.array(ret).T.astype("float")

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def get_feature_names(self, input_features):
        pass


if __name__ == "__main__":
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures as PF

    X = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])
    pf = PF(degree=2, interaction_only=True)
    print(pf.fit_transform(X))
    print(pf.n_input_features_)
    print(pf.n_output_features_)
    # print(pf.get_feature_names(["a", "b", "c"]))
    print(pf.powers_)

    my_pf = PloynomialFeatures(degree=2, interaction_only=True)
    my_pf.fit(X)
    print(my_pf.powers_)
