from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from prefixspan import PrefixSpan

from .base_algo import BaseAlgo


class Rank(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GSPAlgo(BaseAlgo):
    def __init__(self, data):
        self.original_df = data.get_data()
        super().__init__(data)

    def prepare_data(self) -> pd.DataFrame:
        df = self.data.get_data()
        patient_ids = (
            df["id"].astype(str)
            if "id" in df.columns
            else pd.Series([str(i) for i in range(len(df))])
        )
        features = df.columns.drop(["diagnosis", "id"], errors="ignore")
        discretizer = KBinsDiscretizer(
            n_bins=3,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf", # type: ignore
        )
        binned = pd.DataFrame(
            discretizer.fit_transform(df[features]), columns=features, index=df.index
        )
        conditions = [binned == 0.0, binned == 1.0, binned == 2.0]
        choices = [Rank.LOW.value, Rank.MEDIUM.value, Rank.HIGH.value]
        ranked = pd.DataFrame(
            np.select(conditions, choices, default=""),
            columns=binned.columns,
            index=binned.index,
        )
        ranked["diagnosis"] = df["diagnosis"]
        ranked.insert(0, "patient_id", patient_ids)
        for col in features:
            ranked[f"{col}_original"] = df[col]
        return ranked

    def _calculate_z_scores(
        self, row: pd.Series, feature_cols: List[str]
    ) -> Dict[str, float]:
        out = {}
        for col in feature_cols:
            v = row[f"{col}_original"]
            mean = self.original_df[col].mean()
            std = self.original_df[col].std()
            out[col] = abs((v - mean) / std) if std > 0 else 0.0
        return out

    def _generate_sequences_for_strategy(
        self, strategy: str, top_k: int
    ) -> pd.DataFrame:
        df = self.data.get_data()
        patient_ids = (
            df["id"].astype(str)
            if "id" in df.columns
            else pd.Series([str(i) for i in range(len(df))])
        )
        features = df.columns.drop(["diagnosis", "id"], errors="ignore")
        params = {"n_bins": 3, "encode": "ordinal", "strategy": strategy}
        if strategy == "quantile":
            params["quantile_method"] = "averaged_inverted_cdf"
        discretizer = KBinsDiscretizer(**params)
        binned = pd.DataFrame(
            discretizer.fit_transform(df[features]), columns=features, index=df.index
        )
        conditions = [binned == 0.0, binned == 1.0, binned == 2.0]
        choices = [Rank.LOW.value, Rank.MEDIUM.value, Rank.HIGH.value]
        ranked = pd.DataFrame(
            np.select(conditions, choices, default=""),
            columns=binned.columns,
            index=binned.index,
        )
        for col in features:
            ranked[f"{col}_original"] = df[col]
        seq_rows: list[dict[str, str]] = []
        feature_cols = [
            c
            for c in ranked.columns
            if not c.endswith("_original") and c not in ["diagnosis", "patient_id"]
        ]
        for pos, (idx, row) in enumerate(ranked.iterrows()):
            z = self._calculate_z_scores(row, feature_cols)
            ranked_feats = sorted(
                [(c, row[c]) for c in feature_cols],
                key=lambda x: z.get(x[0], 0.0),
                reverse=True,
            )
            top = ranked_feats[:top_k]

            entry = {
                "patient_id": patient_ids.iloc[pos],
                "diagnosis": df["diagnosis"].iloc[pos],
            }

            for i, (name, val) in enumerate(top, 1):
                entry[f"feature_{i}_name"] = name
                entry[f"feature_{i}_value"] = val

            while len(top) < top_k:
                n = len(top) + 1
                entry[f"feature_{n}_name"] = ""
                entry[f"feature_{n}_value"] = ""
                top.append(("", ""))

            seq_rows.append(entry)
        return pd.DataFrame(seq_rows)

    def _convert_df_to_sequences(
        self, df: pd.DataFrame, top_k: int
    ) -> Tuple[List[List[Tuple[str]]], List[List[Tuple[str]]]]:
        mal = []
        ben = []
        for _, row in df.iterrows():
            seq = []
            for i in range(1, top_k + 1):
                name = row[f"feature_{i}_name"]
                val = row[f"feature_{i}_value"]
                if name and val:
                    seq.append((f"{val}_{name}",))
            if row["diagnosis"] == "M":
                mal.append(seq)
            elif row["diagnosis"] == "B":
                ben.append(seq)
        return mal, ben

    def _mine(
        self, sequences: List[List[Tuple[str]]], maxlen: int, min_sup: float
    ) -> List[Tuple[int, List[Tuple[str]]]]:
        if not sequences:
            return []
        sup_count = max(1, int(min_sup * len(sequences)))
        ps = PrefixSpan(sequences)
        ps.minlen = 2
        patterns = ps.frequent(sup_count)
        return [(s, p) for s, p in patterns if len(p) <= maxlen]

    def _patterns_to_df(
        self, patterns: List[Tuple[int, List[Tuple[str]]]], total: int, suffix: str
    ) -> pd.DataFrame:
        if not patterns or total == 0:
            return pd.DataFrame(
                columns=[
                    f"support_count_{suffix}",
                    f"support_percent_{suffix}",
                    "pattern",
                ]
            )
        df = pd.DataFrame(patterns, columns=[f"support_count_{suffix}", "pattern"])
        df[f"support_percent_{suffix}"] = df[f"support_count_{suffix}"] / total
        df["pattern"] = df["pattern"].astype(str)
        return df

    def run(self) -> pd.DataFrame:
        strategies = ["uniform", "quantile", "kmeans"]
        top_k = 10
        min_sup = 0.1
        max_len = 10
        all_res = []
        for strat in strategies:
            seq_df = self._generate_sequences_for_strategy(strat, top_k)
            mal, ben = self._convert_df_to_sequences(seq_df, top_k)
            pm = self._mine(mal, max_len, min_sup)
            pb = self._mine(ben, max_len, min_sup)
            df_m = self._patterns_to_df(pm, len(mal), "m")
            df_b = self._patterns_to_df(pb, len(ben), "b")
            merged = pd.merge(df_m, df_b, on="pattern", how="outer").fillna(0)
            merged["GR_Malignant"] = merged["support_percent_m"] / (
                merged["support_percent_b"] + 1e-12
            )
            merged["GR_Benign"] = merged["support_percent_b"] / (
                merged["support_percent_m"] + 1e-12
            )
            merged["Contrast_Rate"] = np.where(
                (merged["support_percent_m"] > 0) & (merged["support_percent_b"] == 0),
                np.inf,
                np.where(
                    (merged["support_percent_b"] > 0) & (merged["support_percent_m"] == 0),
                    np.inf,
                    merged[["GR_Malignant", "GR_Benign"]].max(axis=1)
                )
            )
            merged["pattern_length"] = merged["pattern"].str.count(r"\(")
            merged["strategy"] = strat
            all_res.append(merged)
        final_cols = [
            "strategy",
            "pattern",
            "pattern_length",
            "support_percent_m",
            "support_percent_b",
            "support_count_m",
            "support_count_b",
            "GR_Malignant",
            "GR_Benign",
            "Contrast_Rate",
        ]
        return pd.concat(all_res, ignore_index=True)[final_cols]
