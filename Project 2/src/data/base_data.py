import pandas as pd
import numpy as np
from pathlib import Path
from config import DATASET_NAME

class Data:
    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self._data: pd.DataFrame = self.process_data()

    def process_data(self) -> pd.DataFrame:
        base_path = Path(__file__).resolve().parent / "raw" / self.folder_name / DATASET_NAME
        df = pd.read_csv(base_path)
        df = df.replace(r'(?i)^nan$', np.nan, regex=True)
        return df.dropna(axis=1, how="all")
    
    def get_data(self) -> pd.DataFrame:
        return self._data

    def save_data(self) -> None:
        output_path = Path(__file__).resolve().parent / "input" / self.folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / DATASET_NAME
        self._data.to_csv(file_path, index=False)
