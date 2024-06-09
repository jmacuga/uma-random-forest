import numpy as np
from pandas.core.api import DataFrame as DataFrame
from datasets.mushrooms import MushroomDataset, Dataset
import pandas as pd


class BreastCancerDataset(MushroomDataset):
    def __init__(self, id: int = 17, path: str = None, label_id: int = 1):
        super().__init__(id=id)

    def __repr__(self):
        return f"BreastCancerDataset() with {len(self)} samples"
    
    def to_numerical(self):
        self.features_map = {col: sorted(self.data[col].unique()) for col in self.data}
        self.labels, self.classes_map = pd.factorize(self.labels)

    def clean(self) -> None:
        self.to_numerical()
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
