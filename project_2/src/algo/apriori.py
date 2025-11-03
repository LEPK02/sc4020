import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any
from .base_algo import BaseAlgo


class AprioriAlgo(BaseAlgo):
    def prepare_data(self) -> List[List[str]]:
        """Prepare data for custom Apriori implementation"""
        df = self.data.get_data()

        symptom_cols = [col for col in df.columns if col.lower().startswith("symptom")]

        # Clean and process data
        df[symptom_cols] = (
            df[symptom_cols].fillna("").astype(str).apply(lambda x: x.str.strip())
        )

        # Convert to transaction format expected by custom Apriori
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in symptom_cols:
                symptom = row[col]
                if symptom and symptom != "nan":  # Filter out empty symptoms
                    transaction.append(symptom)
            if transaction:  # Only add non-empty transactions
                transactions.append(transaction)
        
        return transactions

    @staticmethod
    def min_supp_count(min_supp: float, total_trans: int) -> int:
        """Calculate minimum support count"""
        count = min_supp * total_trans
        return int(count)  # Convert to integer for count

    def get_frequent_itemsets(self, data: List[List[str]], min_supp_count_value: int) -> Dict[frozenset, int]:
        """Get frequent itemsets using custom implementation"""
        symptom_counts = defaultdict(int)

        # Count individual symptoms
        for transaction in data:
            for symptom in transaction:
                symptom_counts[symptom] += 1

        # Filter by minimum support
        frequent = {}
        for symptom, count in symptom_counts.items():
            if count >= min_supp_count_value:
                frequent[frozenset([symptom])] = count

        list_for_rule_generation = frequent.copy()
        templist = frequent
        k = 2

        print(f"Found {len(frequent)} frequent 1 itemsets")

        # Generate larger itemsets
        while templist:
            print(f"Generating {k} itemsets")

            # Create candidate itemsets
            candidates = set()
            itemsets_list = list(templist.keys())

            # Combine itemsets
            for i in range(len(itemsets_list)):
                for j in range(i + 1, len(itemsets_list)):
                    itemset1 = itemsets_list[i]
                    itemset2 = itemsets_list[j]

                    # Check if 2 itemsets can be combined
                    union_set = itemset1.union(itemset2)
                    if len(union_set) == k:
                        candidates.add(union_set)

            if not candidates:
                break

            # Count support for candidates
            candidate_support = {}
            for candidate in candidates:
                count = 0
                for transaction in data:
                    if candidate.issubset(set(transaction)):
                        count += 1

                if count >= min_supp_count_value:
                    candidate_support[candidate] = count

            if not candidate_support:
                break

            print(f"Found {len(candidate_support)} frequent {k}-itemsets")
            list_for_rule_generation.update(candidate_support)
            templist = candidate_support
            k += 1

        return list_for_rule_generation

    def generate_association_rules(self, frequent_itemsets: Dict[frozenset, int], 
                                 min_confidence: float, total_transactions: int) -> List[Dict[str, Any]]:
        """Generate association rules from frequent itemsets"""
        rules = []
        from itertools import combinations

        for itemset, support_count in frequent_itemsets.items():
            if len(itemset) >= 2:  # Generate rules for itemsets with at least 2 items
                items_list = list(itemset)

                # Generate all possible rules for this itemset
                for r in range(1, len(items_list)):
                    for antecedent_items in combinations(items_list, r):
                        antecedent_set = frozenset(antecedent_items)
                        consequent_set = itemset - antecedent_set

                        # Calculate confidence
                        if antecedent_set in frequent_itemsets:
                            antecedent_support = frequent_itemsets[antecedent_set]
                            confidence = support_count / antecedent_support

                            if confidence >= min_confidence:
                                support_ratio = support_count / total_transactions
                                
                                # Calculate lift
                                consequent_support = frequent_itemsets.get(consequent_set, 0) / total_transactions
                                lift = confidence / consequent_support if consequent_support > 0 else 0
                                
                                rules.append({
                                    'antecedents': antecedent_set,
                                    'consequents': consequent_set,
                                    'support': support_ratio,
                                    'confidence': confidence,
                                    'lift': lift,
                                    'antecedent support': antecedent_support / total_transactions,
                                    'consequent support': consequent_support
                                })

        return rules

    def run(
        self,
        min_support: float = 0.05,
        metric: str = "confidence",
        min_threshold: float = 0.7,
    ) -> pd.DataFrame:
        """Run custom Apriori algorithm"""
        
        # For compatibility, map metric to appropriate parameter
        if metric == "confidence":
            min_confidence = min_threshold
        else:
            # You can extend this for other metrics
            min_confidence = min_threshold

        # Run custom Apriori
        total_transactions = len(self.processed_data)
        min_supp_count_value = self.min_supp_count(min_support, total_transactions)

        print(f"Custom Apriori Implementation")
        print(f"Total transactions: {total_transactions}")
        print(f"Minimum support: {min_support:.1%} (≥{min_supp_count_value} transactions)")
        print(f"Minimum confidence: {min_confidence:.1%}")

        # Get frequent itemsets
        frequent_itemsets = self.get_frequent_itemsets(self.processed_data, min_supp_count_value)

        # Generate rules
        rules_list = self.generate_association_rules(frequent_itemsets, min_confidence, total_transactions)

        # Convert to DataFrame for compatibility
        if rules_list:
            rules_df = pd.DataFrame(rules_list)
            
            # Format antecedents and consequents as strings
            rules_df["antecedents"] = rules_df["antecedents"].apply(lambda x: ", ".join(sorted(x)))
            rules_df["consequents"] = rules_df["consequents"].apply(lambda x: ", ".join(sorted(x)))
            
            # Ensure numeric columns
            numeric_cols = ["support", "confidence", "lift", "antecedent support", "consequent support"]
            for col in numeric_cols:
                if col in rules_df.columns:
                    rules_df[col] = pd.to_numeric(rules_df[col], errors="coerce")
                    rules_df[col] = rules_df[col].replace([float("inf"), float("-inf")], 0.0)
                    rules_df[col] = rules_df[col].fillna(0.0)
            
            print(f"Generated {len(rules_df)} association rules")
            return rules_df
        else:
            print("No association rules generated")
            return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
