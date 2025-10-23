import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from .base_algo import BaseAlgo


class AprioriAlgo(BaseAlgo):
    def prepare_data(self) -> pd.DataFrame:
        df = self.data.get_data()

        symptom_cols = [col for col in df.columns if col.lower().startswith("symptom")]

        df[symptom_cols] = (
            df[symptom_cols].fillna("").astype(str).apply(lambda x: x.str.strip())
        )

        all_symptoms = sorted(
            {symptom for col in symptom_cols for symptom in df[col] if symptom}
        )
        encoded_rows = []
        for _, row in df.iterrows():
            encoded_rows.append(
                {symptom: symptom in row.values for symptom in all_symptoms}
            )

        return pd.DataFrame(encoded_rows)

    # TODO: write own implementation of Apriori instead of calling library?
    def run(
        self,
        min_support: float = 0.05,
        metric: str = "confidence",
        min_threshold: float = 0.7,
    ) -> pd.DataFrame:
        frequent_itemsets: pd.DataFrame = apriori(
            self.processed_data, min_support=min_support, use_colnames=True
        )
        rules: pd.DataFrame = association_rules(
            frequent_itemsets, metric=metric, min_threshold=min_threshold
        )

        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

        numeric_cols = [
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "representativity",
            "leverage",
            "conviction",
            "zhangs_metric",
            "jaccard",
            "certainty",
            "kulczynski",
        ]
        for col in numeric_cols:
            if col in rules.columns:
                rules[col] = pd.to_numeric(rules[col], errors="coerce")
                rules[col] = rules[col].replace([float("inf"), float("-inf")], 0.0)
                rules[col] = rules[col].fillna(0.0)

        return rules
