from pathlib import Path
from typing import Any, Dict, List, cast

import joblib
import numpy as np
import pandas as pd
from data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import MultiLabelBinarizer


class PredictionModel:
    def __init__(self, data: Data) -> None:
        self.model_path = Path(__file__).resolve().parent / "prediction_model.joblib"
        self.df = data.get_data()
        self.mlb = MultiLabelBinarizer()
        self.model = RandomForestClassifier()
        self.metrics: Dict[str, Any] | None = None
        self.best_params: dict | None = None
        if not self.model_path.exists():
            self.prepare_data()
            self.train_model()
            self.save_model()
        else:
            self.load_model()

    def prepare_data(self) -> pd.DataFrame:
        symptom_cols = [c for c in self.df.columns if c.lower().startswith("symptom")]
        self.df["Symptoms"] = self.df[symptom_cols].apply(
            lambda x: [
                s.strip()
                for s in x.dropna().tolist()
                if isinstance(s, str) and s.strip()
            ],
            axis=1,
        )
        self.X = np.array(self.mlb.fit_transform(self.df["Symptoms"]), dtype=int)
        self.y = np.array(self.df["Disease"])
        return self.df

    def train_model(self, n_repeats: int = 5) -> None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)

        metrics_acc, metrics_prec, metrics_rec, metrics_f1 = [], [], [], []
        metrics_reports = []
        metrics_cm = []

        best_estimators = []

        for _ in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, stratify=self.y, random_state=None
            )

            grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=cv,
                n_jobs=5,
                scoring="f1_weighted",
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            best_estimators.append(best_model)
            y_pred = best_model.predict(X_test)

            p, r, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )
            metrics_acc.append(accuracy_score(y_test, y_pred))
            metrics_prec.append(p)
            metrics_rec.append(r)
            metrics_f1.append(f1)
            metrics_reports.append(cast(Dict[str, Any], classification_report(y_test, y_pred, output_dict=True, zero_division=0)))
            metrics_cm.append(confusion_matrix(y_test, y_pred))

        best_idx = np.argmax(metrics_acc)
        self.model = best_estimators[best_idx]
        self.best_params = self.model.get_params()
        self.metrics = {
            "accuracy": float(np.mean(metrics_acc)),
            "precision": float(np.mean(metrics_prec)),
            "recall": float(np.mean(metrics_rec)),
            "f1_score": float(np.mean(metrics_f1)),
            "classification_report": metrics_reports[best_idx],
            "confusion_matrix": metrics_cm[best_idx].tolist(),
        }


    def save_model(self) -> None:
        data = {
            "model": self.model,
            "mlb": self.mlb,
            "metrics": self.metrics,
            "best_params": self.best_params,
        }
        joblib.dump(data, self.model_path)

    def load_model(self) -> None:
        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.mlb = data["mlb"]
        self.metrics = data.get("metrics")
        self.best_params = data.get("best_params")

    def predict(self, symptoms: List[str]) -> pd.DataFrame:
        input_vector = pd.DataFrame(
            [0] * len(self.mlb.classes_), index=self.mlb.classes_
        ).T
        for s in symptoms:
            if s in self.mlb.classes_:
                input_vector[s] = 1
        probs = self.model.predict_proba(input_vector)
        return (
            pd.DataFrame(probs, columns=self.model.classes_)
            .T.rename(columns={0: "probability"})
            .sort_values(by="probability", ascending=False)
        )
