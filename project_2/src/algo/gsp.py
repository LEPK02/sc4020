from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from .base_algo import BaseAlgo


class Rank(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GSPAlgo(BaseAlgo):
    def __init__(self, data):
        """
        Args:
            data: Data object
        """
        # Store original numerical data for z-score calculation before calling super().__init__
        self.original_df = data.get_data()
        super().__init__(data)
    
    def prepare_data(self) -> pd.DataFrame:
        df: pd.DataFrame = self.data.get_data()
        patient_ids: pd.Series = (
            df["id"].astype(str) if "id" in df.columns else pd.Series([str(i) for i in range(len(df))])
        )
        features: pd.Index = df.columns.drop(["diagnosis", "id"], errors="ignore")
        
        # Use quantile as default for prepare_data (will generate all strategies in run())
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
        # Store original numerical values alongside discretized ranks
        for col in features:
            ranked[f"{col}_original"] = df[col]
        return ranked

    def _calculate_z_scores(self, patient_row: pd.Series, feature_cols: pd.Index) -> dict[str, float]:
        """Calculate absolute z-scores for patient's features"""
        z_scores = {}
        for col in feature_cols:
            original_col = f"{col}_original"
            if original_col in patient_row.index:
                value = patient_row[original_col]
                # Calculate z-score: (value - population_mean) / population_std
                pop_mean = self.original_df[col].mean()
                pop_std = self.original_df[col].std()
                if pop_std > 0:
                    z_score = abs((value - pop_mean) / pop_std)
                else:
                    z_score = 0.0
                z_scores[col] = z_score
        return z_scores

    def _generate_sequences_for_strategy(self, strategy: str, max_len: int = 3) -> pd.DataFrame:
        """Generate sequences for a specific discretization strategy"""
        # Prepare data with the specified strategy
        df: pd.DataFrame = self.data.get_data()
        patient_ids: pd.Series = (
            df["id"].astype(str) if "id" in df.columns else pd.Series([str(i) for i in range(len(df))])
        )
        features: pd.Index = df.columns.drop(["diagnosis", "id"], errors="ignore")
        
        # Create discretizer with selected strategy
        discretizer_params = {
            "n_bins": 3,
            "encode": "ordinal",
            "strategy": strategy,
        }
        
        # Add quantile_method only for quantile strategy
        if strategy == "quantile":
            discretizer_params["quantile_method"] = "averaged_inverted_cdf"
        
        discretizer: KBinsDiscretizer = KBinsDiscretizer(**discretizer_params)
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
        
        # Store original values for z-score calculation
        for col in features:
            ranked[f"{col}_original"] = df[col]
        
        # Generate sequences with z-score ranking
        sequences: list[dict] = []
        feature_cols = [col for col in ranked.columns 
                       if not col.endswith('_original') and col not in ['diagnosis', 'patient_id']]
        
        for pos_idx, (idx, row) in enumerate(ranked.iterrows()):
            # Calculate z-scores per patient and rank features by z-score
            scores = self._calculate_z_scores(row, pd.Index(feature_cols))
            # Sort by z-score descending (higher deviation = more important)
            ranked_features = sorted(
                [(col, row[col]) for col in feature_cols if col in scores],
                key=lambda x: scores.get(x[0], 0.0),
                reverse=True
            )
            
            # Take top max_len features
            seq: list[tuple[str, str]] = ranked_features[:max_len]
            
            entry: dict = {
                "patient_id": patient_ids.iloc[pos_idx],
                "diagnosis": df["diagnosis"].iloc[pos_idx],
            }
            for i, (name, value) in enumerate(seq, 1):
                entry[f"feature_{i}_name"] = name
                entry[f"feature_{i}_value"] = value
            
            # Fill remaining features with empty if needed
            while len(seq) < max_len:
                i = len(seq) + 1
                entry[f"feature_{i}_name"] = ""
                entry[f"feature_{i}_value"] = ""
                seq.append(("", ""))
            
            sequences.append(entry)
        
        return pd.DataFrame(sequences)

    def run(self, max_len: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Generate sequences for all three discretization strategies using z-score ranking.
        
        Args:
            max_len: Maximum sequence length (number of features)
        
        Returns:
            Dictionary with keys "uniform", "quantile", "kmeans" containing DataFrames
        """
        results = {}
        strategies = ["uniform", "quantile", "kmeans"]
        
        for strategy in strategies:
            results[strategy] = self._generate_sequences_for_strategy(strategy, max_len)
        
        return results
