from enum import Enum


DATASET_NAME = "dataset.csv"


class TaskName(Enum):
    DISEASE_SYMPTOMS = "[Task 1] Disease Symptoms"
    CANCER_FEATURES = "[Task 2] Cancer Features"
    DISEASE_PREDICTION = "[Task 3] Disease Prediction"
