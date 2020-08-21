from abc import abstractmethod
import pandas as pd
import numpy as np


def _to_numpy(x):
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array(x)
    raise TypeError("Input should be pandas dataframe or numpy array!")


class BasePreprocessor:
    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def fit_transform(self, x):
        pass

    def get_params(self):
        return self.__dict__

    def set_params(self, **params):
        for k, v in params.items():
            if k in self.__dict__:
                self.__dict__[k] = v

    def _check_type(self, x):
        if not isinstance(x, (pd.DataFrame, np.ndarray, list)):
            raise TypeError("{}'s input should be pandas dataframe or numpy array!".format(self.__class__.__name__))
        return True

    @staticmethod
    def _to_numpy(x):
        return _to_numpy(x)


BaseImputer = BasePreprocessor


BaseDecompositor = BasePreprocessor


class BaseModel:
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def get_params(self):
        return self.__dict__

    def set_params(self, **params):
        for k, v in params.items():
            if k in self.__dict__.keys():
                print(k, v)
                self.__dict__[k] = v

    @staticmethod
    def _to_numpy(x):
        return _to_numpy(x)