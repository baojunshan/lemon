from stlearn.base import BasePreprocessor


class FunctionError(Exception):
    pass


class FunctionTransformer(BasePreprocessor):
    def __init__(self,
                 func=None,
                 inv_func=None,
                 check_inv=True,
                 kw_args: dict = None,
                 inv_kw_args: dict = None
                 ):
        self.func = func
        self.inv_func = inv_func
        self.check_inv = check_inv
        self.kw_args = kw_args if kw_args else dict()
        self.inv_kw_args = inv_kw_args if inv_kw_args else dict()

    def __check_inv(self, x):
        if not self.inv_func:
            return
        trans_twice_x = self.inv_func(self.func(x, **self.kw_args), **self.inv_kw_args)
        if not (trans_twice_x.round(12) == x).all():
            raise FunctionError("Inverse function could not revert to the original data.")

    def fit(self, x):
        self._check_type(x)
        if self.check_inv:
            self.__check_inv(x)
        return self

    def transform(self, x):
        x = self._to_numpy(x)
        if self.check_inv:
            self.__check_inv(x)
        return self.func(x, **self.kw_args)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if not self.inv_func:
            raise AttributeError("inv_func should not be None")
        x = self._to_numpy(x)
        if self.check_inv:
            self.__check_inv(x)
        return self.inv_func(x, **self.kw_args)
