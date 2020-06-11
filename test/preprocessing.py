from stlearn.processing.preprocessing import Binarizer, FunctionTransformer


def call_name(func):
    def wrapped(*args, **kwargs):
        print(f"[func]: {func.__name__}")
        func()
        print("\n")

    return wrapped


@call_name
def check_binarizer():
    x = [[1., -1., 2.],
         [2., 0., 0.],
         [0., 1., -1.]]
    binarizer = Binarizer()
    y = binarizer.fit_transform(x)
    print(y)
    print(binarizer.get_params())


@call_name
def check_function_transformer():
    x = [[0, 1], [2, 3]]

    def ft_func(x):
        return x + 1

    def ft_inv_func(x):
        return x - 1

    function_transformer = FunctionTransformer(func=ft_func, inv_func=ft_inv_func)
    x1 = function_transformer.fit_transform(x)
    print(x1, "\n", function_transformer.get_params())

    import numpy as np

    function_transformer.set_params(func=np.sqrt, inv_func=np.square)
    x2 = function_transformer.fit_transform(x)
    print(x2, "\n", function_transformer.get_params())


if __name__ == "__main__":
    check_binarizer()
    check_function_transformer()
