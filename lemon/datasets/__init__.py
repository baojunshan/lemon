import pickle
from pathlib import PurePath

__all__ = [
    "load_iris",
    "load_boston",
    "load_breast_cancer",
    "load_wine",
    "load_titanic"
]

CURR_FOLDER_PATH = PurePath(__file__).parent


def _load_data(filename, x_y):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    if not x_y:
        return data
    return data["data"], data["target"]


def load_iris(x_y=False):
    return _load_data(str(CURR_FOLDER_PATH / "iris.pkl"), x_y)


def load_boston(x_y=False):
    return _load_data(str(CURR_FOLDER_PATH / "boston.pkl"), x_y)


def load_breast_cancer(x_y=False):
    return _load_data(str(CURR_FOLDER_PATH / "breast_cancer.pkl"), x_y)


def load_wine(x_y=False):
    return _load_data(str(CURR_FOLDER_PATH / "wine.pkl"), x_y)


def load_titanic(x_y=False):
    return _load_data(str(CURR_FOLDER_PATH / "titanic.pkl"), x_y)
