import re
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

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

    def min_supp_count(self, min_supp: float, total_trans: int) -> float:
        return min_supp * total_trans

    def get_frequent_itemsets(
        self, data: List[List[str]], min_supp_count_value: float
    ) -> Dict[frozenset[str], int]:
        symptom_counts: Dict[str, int] = defaultdict(int)
        for i in data:
            for j in i:
                symptom_counts[j] += 1
        frequent: Dict[frozenset[str], int] = {}
        for i in symptom_counts:
            count = symptom_counts[i]
            if count >= min_supp_count_value:
                frequent[frozenset([i])] = count
        list_for_rule_generation: Dict[frozenset[str], int] = frequent.copy()
        templist: Dict[frozenset[str], int] = frequent
        k = 2
        while templist:
            candidates: Set[frozenset[str]] = set()
            itemsets_list = list(templist.keys())
            for i in range(len(itemsets_list)):
                for j in range(i + 1, len(itemsets_list)):
                    union_set = itemsets_list[i].union(itemsets_list[j])
                    if len(union_set) == k:
                        candidates.add(union_set)
            if not candidates:
                break
            candidate_support: Dict[frozenset[str], int] = {}
            for candidate in candidates:
                count = 0
                for transaction in data:
                    if candidate.issubset(set(transaction)):
                        count += 1
                if count >= min_supp_count_value:
                    candidate_support[candidate] = count
            if not candidate_support:
                break
            list_for_rule_generation.update(candidate_support)
            templist = candidate_support
            k += 1
        return list_for_rule_generation

    def generate_association_rules(
        self,
        frequent_itemsets: Dict[frozenset[str], int],
        min_confidence: float,
        total_transactions: int,
    ) -> List[Dict[str, Any]]:
        rules_dict: Dict[frozenset[str], Dict[str, Any]] = {}

        for itemset, support_count in frequent_itemsets.items():
            if len(itemset) >= 2:
                items_list = list(itemset)
                for r in range(1, len(items_list)):
                    for antecedent_items in combinations(items_list, r):
                        antecedent_set = frozenset(antecedent_items)
                        consequent_set = itemset - antecedent_set
                        if antecedent_set in frequent_itemsets:
                            antecedent_support = frequent_itemsets[antecedent_set]
                            confidence = support_count / antecedent_support
                            if confidence >= min_confidence:
                                key = frozenset(antecedent_set.union(consequent_set))
                                existing_rule = rules_dict.get(key)
                                if not existing_rule or confidence > existing_rule["confidence"]:
                                    rules_dict[key] = {
                                        "antecedent": " & ".join(sorted(antecedent_set)),
                                        "consequent": " & ".join(sorted(consequent_set)),
                                        "support": support_count / total_transactions,
                                        "support_count": support_count,
                                        "confidence": confidence,
                                    }

        rules_list = [
            {**rule, "rule_id": i + 1} for i, rule in enumerate(rules_dict.values())
        ]
        return rules_list


    def get_unique_itemsets_proper(self, df: pd.DataFrame) -> List[str]:
        seen: Set[frozenset] = set()
        unique_itemsets: List[str] = []
        for _, row in df.iterrows():
            itemset = frozenset([row["antecedent"], row["consequent"]])
            if itemset not in seen:
                seen.add(itemset)
                sorted_items = sorted(list(itemset))
                unique_itemsets.append(f"{sorted_items[0]} & {sorted_items[1]}")
        return unique_itemsets

    def extract_from_string(self, itemset_strings: List[str]) -> Set[Tuple[str, str]]:
        unique_itemsets: Set[Tuple[str, str]] = set()
        for itemset_str in itemset_strings:
            matches = re.findall(r"'([^']*)'", itemset_str)
            if len(matches) >= 2:
                item1, item2 = matches[0], matches[1]
                sorted_pair = tuple(sorted([item1, item2]))
                unique_itemsets.add(sorted_pair)
        return unique_itemsets

    def find_diseases_for_symptom_pairs(
        self, df: pd.DataFrame, symptom_pairs: Set[Tuple[str, str]]
    ) -> Dict[str, List[str]]:
        df_clean = df.copy()
        symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
        for col in symptom_cols:
            df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        symptom_mapping = {
            'abdominal_pain': 'stomach_pain',
            'belly_pain': 'stomach_pain'
        }
        for col in symptom_cols:
            df_clean[col] = df_clean[col].replace(symptom_mapping)
            
        disease_symptom_mapping: Dict[str, List[str]] = {}
        for symptom_pair in symptom_pairs:
            symptom1, symptom2 = symptom_pair
            matching_diseases: List[str] = []
            for _, row in df_clean.iterrows():
                disease = row["Disease"]
                symptoms = {str(s).strip() for s in row[1:] if isinstance(s, str) and s.strip()}
                if symptom1 in symptoms and symptom2 in symptoms:
                    matching_diseases.append(disease)
            if matching_diseases:
                disease_symptom_mapping[f"{symptom1} & {symptom2}"] = list(
                    set(matching_diseases)
                )
        return disease_symptom_mapping

    def run(self) -> pd.DataFrame:
        min_support = 0.15
        min_confidence = 0.5
        data_list: List[List[str]] = [
            [str(symptom) for symptom, present in row.items() if present]
            for _, row in self.processed_data.iterrows()
        ]
        
        total_count = len(data_list)
        min_supp_count_value = self.min_supp_count(min_support, total_count)
        frequent_itemsets = self.get_frequent_itemsets(data_list, min_supp_count_value)
        rules = self.generate_association_rules(
            frequent_itemsets, min_confidence, total_count
        )

        return pd.DataFrame(rules).drop(columns=["rule_id"], errors="ignore")