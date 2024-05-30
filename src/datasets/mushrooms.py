import numpy as np
from pandas.core.api import DataFrame as DataFrame
from utils.types import Dataset
import pandas as pd
from ucimlrepo import fetch_ucirepo


class MushroomDataset(Dataset):
    def __init__(self, path: str = None):
        super().__init__()
        if path is None:
            mushroom = fetch_ucirepo(id=73)

            self.data = np.array(mushroom.data.features)
            self.labels = np.array(mushroom.data.targets)
        else:
            mushrooms = pd.read_csv(path, header=None)
            self.data = mushrooms.iloc[:, 1:]
            self.labels = mushrooms.iloc[:, 0]

        self.features_map = {}
        self.classes_map = {}

    def __repr__(self):
        return f"ShroomsDataset() with {len(self)} samples"

    def to_numerical(self):
        for col in self.data.columns:
            dummies = pd.get_dummies(self.data[col], prefix=col)
            self.data = pd.concat([self.data.drop(col, axis=1), dummies], axis=1)
            self.features_map[col] = dummies.columns

        self.labels, self.classes_map = pd.factorize(self.labels)

    def clean(self) -> None:
        self.to_numerical()
        self.data = np.array(self.data, dtype=int)
        self.labels = np.array(self.labels, dtype=int)
