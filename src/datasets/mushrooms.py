import numpy as np
from pandas.core.api import DataFrame as DataFrame
from utils.types import Dataset

from ucimlrepo import fetch_ucirepo


class MushroomDataset(Dataset):
    def __init__(self):
        super().__init__()
        mushroom = fetch_ucirepo(id=73)
        self.data = np.array(mushroom.data.features)
        self.labels = np.array(mushroom.data.targets)

    def __repr__(self):
        return f"ShroomsDataset() with {len(self)} samples"

    def clean(self) -> DataFrame:
        return super().clean()
