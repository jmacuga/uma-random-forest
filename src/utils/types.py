from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


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

    @abstractmethod
    def load(self) -> np.array:
        pass

    @abstractmethod
    def split_(self) -> tuple[np.array, np.array, np.array, np.array]:
        pass


def Experiment(ABC):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.data = dataset.data
        self.labels = dataset.labels

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def clean(self):
        pass
