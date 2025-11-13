from pathlib import Path
from typing import List

import joblib
import pandas as pd
from data import Data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


class PredictionModel:
    def __init__(self, data: Data) -> None:
        self.model_path = Path(__file__).resolve().parent / "prediction_model.joblib"
        self.df = data.get_data()
        self.mlb = MultiLabelBinarizer()
        self.model = LogisticRegression(max_iter=1000)
        self.accuracy: float | None = None
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
            lambda x: [s.strip() for s in x.dropna().tolist() if isinstance(s, str) and s.strip()],
            axis=1,
        )
        self.X = self.mlb.fit_transform(self.df["Symptoms"])
        self.y = self.df["Disease"]
        return self.df

    def train_model(self) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "saga", "newton-cg"],
            "penalty": ["l2"],
        }

        grid = GridSearchCV(
            LogisticRegression(max_iter=1000),
            param_grid,
            cv=3,
            n_jobs=-1,
            scoring="accuracy",
        )
        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.accuracy = float(self.model.score(X_test, y_test))

        print(
            f"Model trained with accuracy: {self.accuracy:.2%}, "
            f"best params: {self.best_params}"
        )

    def save_model(self) -> None:
        data = {
            "model": self.model,
            "mlb": self.mlb,
            "accuracy": self.accuracy,
            "best_params": self.best_params,
        }
        joblib.dump(data, self.model_path)
        print(f"Model saved to: {self.model_path.name}")

    def load_model(self) -> None:
        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.mlb = data["mlb"]
        self.accuracy = data.get("accuracy")
        self.best_params = data.get("best_params")
        print(
            f"Loaded model from: {self.model_path.name} "
            f"(accuracy: {self.accuracy:.2%}, best_params: {self.best_params})"
        )

    def predict(self, symptoms: List[str]) -> pd.DataFrame:
        input_vector = pd.DataFrame([0] * len(self.mlb.classes_), index=self.mlb.classes_).T
        for s in symptoms:
            if s in self.mlb.classes_:
                input_vector[s] = 1
        probs = self.model.predict_proba(input_vector)
        return (
            pd.DataFrame(probs, columns=self.model.classes_)
            .T.rename(columns={0: "probability"})
            .sort_values(by="probability", ascending=False)
        )
