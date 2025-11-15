from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from config import DATASET_NAME
from data import Data


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

        result = self.run()

        # Handle dictionary of DataFrames (for GSPAlgo)
        if isinstance(result, dict):
            for strategy, df in result.items():
                file_path = (
                    output_path / f"dataset_{strategy}_features.csv"
                )
                df.to_csv(file_path, index=False)
        # Handle single DataFrame (for other algorithms)
        elif isinstance(result, pd.DataFrame):
            base_name = DATASET_NAME.rsplit(".", 1)[0]
            extension = DATASET_NAME.rsplit(".", 1)[1] if "." in DATASET_NAME else "csv"
            file_path = output_path / f"{base_name}_features.{extension}"
            result.to_csv(file_path, index=False)
        else:
            raise TypeError("run() must return pd.DataFrame")
