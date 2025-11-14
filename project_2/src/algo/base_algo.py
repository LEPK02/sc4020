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

    def save_data(self, top_k: int = None, **kwargs) -> None:
        """
        Save the results of run() to CSV files.
        
        Args:
            top_k: Optional number of top-K features (passed to run() if supported)
        """
        output_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "output"
            / self.data.folder_name
        )
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use default top_k=3 if not provided
        default_top_k = top_k if top_k is not None else 3
        
        # Call run() with top_k if provided, otherwise use default
        if top_k is not None:
            result = self.run(top_k=top_k, **kwargs)
        else:
            result = self.run(**kwargs)
        
        # Handle dictionary of DataFrames (for GSPAlgo)
        if isinstance(result, dict):
            for strategy, df in result.items():
                file_path = output_path / f"dataset_{strategy}_{default_top_k}features.csv"
                df.to_csv(file_path, index=False)
        # Handle single DataFrame (for other algorithms)
        elif isinstance(result, pd.DataFrame):
            base_name = DATASET_NAME.rsplit('.', 1)[0]
            extension = DATASET_NAME.rsplit('.', 1)[1] if '.' in DATASET_NAME else 'csv'
            file_path = output_path / f"{base_name}_{default_top_k}features.{extension}"
            result.to_csv(file_path, index=False)
        else:
            raise TypeError(f"run() must return pd.DataFrame or dict[str, pd.DataFrame], got {type(result)}")
