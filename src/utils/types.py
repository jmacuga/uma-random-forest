from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Classifier(ABC):
    def __init__(self):
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def fit(self, X: np.array, y: np.array):
        raise NotImplementedError("Method fit not implemented")

    @abstractmethod
    def predict(self, X: np.array) -> np.array:
        raise NotImplementedError("Method predict not implemented")


class Dataset(ABC):
    def __init__(self, path: str = None):
        self.path = path
        self.data = None
        self.labels = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Dataset(path={self.path})"

    @abstractmethod
    def clean(self) -> pd.DataFrame:
        pass

    def split(self, test_size: float, random_state: int = None) -> tuple[np.array, np.array, np.array, np.array]:
        return train_test_split(self.data, self.labels, test_size=test_size, random_state=random_state)
