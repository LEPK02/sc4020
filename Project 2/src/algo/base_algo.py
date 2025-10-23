from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from config import DATASET_NAME
from data.base_data import Data


class BaseAlgo(ABC):
    def __init__(self, data: Data):
        self.data = data
        self.processed_data = self.prepare_data()

    @abstractmethod
    def prepare_data(self) -> pd.DataFrame: ...

    @abstractmethod
    def run(self) -> pd.DataFrame: ...

    def save_data(self) -> None:
        output_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "output"
            / self.data.folder_name
        )
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / DATASET_NAME
        result = self.run()
        result.to_csv(file_path, index=False)
