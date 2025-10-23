from collections import defaultdict

def min_supp_count(min_supp, total_trans):
    count = min_supp * total_trans  # min_supp is percentage (e.g., 0.05 for 5%)
    return count

def get_frequent_itemsets(data, min_supp_count_value):

    symptom_counts = defaultdict(int) 
    
    for i in data: #loop throgh the list and count
        for j in i:
            symptom_counts[j] += 1 
    
    # Filter by minimum support
    frequent = {}
    for i in symptom_counts: 
        count = symptom_counts[i]  
        if count >= min_supp_count_value:
            frequent[frozenset([i])] = count

    
    list_for_rule_generation = frequent.copy()
    templist = frequent
    k = 2
    
    print(f"Found {len(frequent)} frequent 1 itemsets")
    
    #Generate larger itemsets
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
                
                # Check if 2 itemsets have something in common then fuse them
                union_set = itemset1.union(itemset2)
                if len(union_set) == k:
                    candidates.add(union_set)
        
        if not candidates:
            break
        
        # Count support
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

def generate_association_rules(frequent_itemsets, min_confidence, total_transactions):

    rules = []
    
    for itemset, support_count in frequent_itemsets.items():
        if len(itemset) >= 2:  # generate rules for itemsets with at least 2 items
            items_list = list(itemset)
            
            # Generate all possible rules for this itemset
            # For itemset {A,B,C}, generate rules eg a-bc, b-ac ....
            from itertools import combinations
            
            # Generate all non-empty proper subsets as antecedents
            for r in range(1, len(items_list)):
                for antecedent_items in combinations(items_list, r):
                    antecedent_set = frozenset(antecedent_items)
                    consequent_set = itemset - antecedent_set
                    
                    # Calculate confidence
                    if antecedent_set in frequent_itemsets:
                        antecedent_support = frequent_itemsets[antecedent_set]
                        confidence = support_count / antecedent_support
                        
                        if confidence >= min_confidence:
                            rules.append({
                                'rule_id': len(rules) + 1,
                                'antecedent': antecedent_set,
                                'consequent': consequent_set,
                                'support': support_count / total_transactions,
                                'support_count': support_count,
                                'confidence': confidence
                            })
    
    return rules

def apriori(data, min_supp, confidence):

    total_count = len(data)
    min_supp_count_value = min_supp_count(min_supp, total_count)
    
    print(f"Apriori")
    print(f"Total transactions: {total_count}")
    print(f"Minimum support: {min_supp:.1%} (≥{min_supp_count_value})")
    print(f"Minimum confidence: {confidence:.1%}")
    
    frequent_itemsets = get_frequent_itemsets(data, min_supp_count_value)
    
    itemset_sizes = {}
    for itemset in frequent_itemsets.keys():
        size = len(itemset)
        itemset_sizes[size] = itemset_sizes.get(size, 0) + 1
    
    for size, count in sorted(itemset_sizes.items()):
        print(f"  {size}-itemsets: {count}")
    
    print(f"\nSTEP 2: GENERATING ASSOCIATION RULES...")
    rules = generate_association_rules(frequent_itemsets, confidence, total_count)
    
    print(f"Generated {len(rules)} association rules")
    
    return frequent_itemsets, rules
