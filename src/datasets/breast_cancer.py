import numpy as np
from pandas.core.api import DataFrame as DataFrame
from datasets.mushrooms import MushroomDataset


class BreastCancerDataset(MushroomDataset):
    def __init__(self, id: int = 14, path: str = None):
        super().__init__(id=id, path=path)

    def __repr__(self):
        return f"BreastCancerDataset() with {len(self)} samples"
