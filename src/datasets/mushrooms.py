from utils.types import Dataset


class ShroomsDataset(Dataset):
    def __init__(self, path: str):
        super().__init__(path)
