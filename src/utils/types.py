from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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

    def split(
        self, test_size: float, random_state: int = None
    ) -> tuple[np.array, np.array, np.array, np.array]:
        return train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state
        )


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
