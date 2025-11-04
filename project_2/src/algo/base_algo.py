from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

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
    def run(self) -> Union[pd.DataFrame, dict[str, pd.DataFrame], Any]: ...

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
                file_path = output_path / f"dataset_{strategy}.csv"
                df.to_csv(file_path, index=False)
        # Handle single DataFrame (for other algorithms)
        elif isinstance(result, pd.DataFrame):
            file_path = output_path / DATASET_NAME
            result.to_csv(file_path, index=False)
        else:
            raise TypeError(f"run() must return pd.DataFrame or dict[str, pd.DataFrame], got {type(result)}")
