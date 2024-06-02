import numpy as np
from pandas.core.api import DataFrame as DataFrame
from utils.types import Dataset
import pandas as pd
from ucimlrepo import fetch_ucirepo


class AirlineCSDataset(Dataset):
    def __init__(self, path: str = None):
        super().__init__()
        airline_cs = pd.read_csv(path)
        self.data = airline_cs.iloc[:, 1:]
        self.labels = airline_cs.iloc[:, 0]

        self.features_map = {}
        self.classes_map = {}

    def __repr__(self):
        return f"Airline Customer Satisfaction Dataset with {len(self)} samples"

    def to_numerical(self):
        columns = ["Customer Type", "Type of Travel", "Class", "Seat comfort"]
        for col in columns:
            dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True, dtype=np.int8)

            self.data = pd.concat([self.data.drop(col, axis=1), dummies], axis=1)
            self.features_map[col] = dummies.columns

        self.labels, self.classes_map = pd.factorize(self.labels)

    def clean(self) -> None:
        self.to_numerical()
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
