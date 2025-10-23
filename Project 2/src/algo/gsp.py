from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from .base_algo import BaseAlgo


class Rank(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GSPAlgo(BaseAlgo):
    def prepare_data(self) -> pd.DataFrame:
        df: pd.DataFrame = self.data.get_data()
        patient_ids: pd.Series = (
            df["id"] if "id" in df.columns else pd.Series(range(len(df)))
        )
        features: pd.Index = df.columns.drop(["diagnosis", "id"], errors="ignore")
        discretizer: KBinsDiscretizer = KBinsDiscretizer(
            n_bins=3,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )
        binned: pd.DataFrame = pd.DataFrame(
            discretizer.fit_transform(df[features]), columns=features, index=df.index
        )
        conditions = [binned == 0.0, binned == 1.0, binned == 2.0]
        choices = [Rank.LOW.value, Rank.MEDIUM.value, Rank.HIGH.value]
        ranked: pd.DataFrame = pd.DataFrame(
            np.select(conditions, choices, default=""),
            columns=binned.columns,
            index=binned.index,
        )
        ranked["diagnosis"] = df["diagnosis"]
        ranked.insert(0, "patient_id", patient_ids)
        return ranked

    def run(self, max_len: int = 3) -> pd.DataFrame:
        sequences: list[dict] = []
        for _, row in self.processed_data.iterrows():
            features: pd.Series = row.drop(["diagnosis", "patient_id"])
            ordered: list[tuple[str, str]] = sorted(
                [(str(k), str(v)) for k, v in features.items()],
                key=lambda x: {
                    Rank.LOW.value: 0,
                    Rank.MEDIUM.value: 1,
                    Rank.HIGH.value: 2,
                }[x[1]],
                reverse=True,
            )
            seq: list[tuple[str, str]] = ordered[:max_len]
            entry: dict = {
                "patient_id": row["patient_id"],
                "diagnosis": row["diagnosis"],
            }
            for i, (name, value) in enumerate(seq, 1):
                entry[f"feature_{i}_name"] = name
                entry[f"feature_{i}_value"] = value
            sequences.append(entry)
        return pd.DataFrame(sequences)

    # TODO: make data analysis table and graphs