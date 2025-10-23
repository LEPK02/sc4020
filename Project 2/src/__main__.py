from algo.apriori import AprioriAlgo
from algo.gsp import GSPAlgo
from data.data import cancer_data, disease_symptoms_data

if __name__ == "__main__":
    apriori_task = AprioriAlgo(disease_symptoms_data)
    # rules = apriori_task.run(min_support=0.1)
    # print(rules.head())
    apriori_task.save_data()

    gsp_task = GSPAlgo(cancer_data)
    # sequences_df = gsp_task.run(max_len=3)
    # print("Sample Sequences:\n", sequences_df.head())
    gsp_task.save_data()
