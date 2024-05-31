import numpy as np
from pandas.core.api import DataFrame as DataFrame
from utils.types import Dataset
import pandas as pd
from ucimlrepo import fetch_ucirepo


class CovertypeDataset(Dataset):
    def __init__(self, path: str = None):
        super().__init__()
        if path is None:
            covertypes = fetch_ucirepo(id=31)

            self.data = np.array(covertypes.data.features)
            self.labels = np.array(covertypes.data.targets)
        else:
            covertypes = pd.read_csv(path, header=None)
            self.data = covertypes.iloc[:, :-1]
            self.labels = covertypes.iloc[:, -1]

        self.features_map = {}
        self.classes_map = {}
        self.labels_names = (
            "Spruce/Fir",
            "Lodgepole Pine",
            "Ponderosa Pine",
            "Cottonwood/Willow",
            "Aspen",
            "Douglas-fir",
            "Krummholz",
        )

    def __repr__(self):
        return f"CovertypeDataset with {len(self)} samples"

    def clean(self) -> None:
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
