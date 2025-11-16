import time
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from algo import PredictionModel
from config import TaskName
from data import disease_symptoms_data


class PredictionPage:
    def __init__(self):
        self.title = TaskName.DISEASE_PREDICTION.value
        self.display_title = "[Task 3] Disease Prediction using ML"
        with st.spinner("Loading model and data…"):
            self.task = PredictionModel(disease_symptoms_data)
            self.get_symptom_options()

    def render(self):
        st.title(self.display_title)

        display_options = [self.format_symptom_display(s) for s in self.symptom_options]
        selected_display = st.multiselect(
            "Select your symptoms:",
            options=display_options,
            help="You can select one or more symptoms to get your diagnosis.",
        )

        selected_symptoms = [self.symptom_map[s] for s in selected_display]

        if st.button("🩺 Diagnose Me!"):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
                return

            with st.spinner("Analyzing symptoms and predicting disease probabilities…"):
                results = self.task.predict(selected_symptoms)

            top_results = results.head(5)
            st.subheader("Top Possible Diseases")

            st.write("5 diseases you are most likely to have:")
            top_results_display = top_results.reset_index()
            top_results_display.columns = ["Disease Name", "Probability of Disease"]
            st.dataframe(
                top_results_display.style.format({"Probability of Disease": "{:.2%}"}),
                width="stretch",
                hide_index=True,
            )

            st.subheader("Model Performance Metrics")
            if self.task.metrics is not None:
                self.show_performance_metrics()

            status_placeholder = st.empty()
            status_placeholder.success("Diagnosis complete!")
            time.sleep(3)
            status_placeholder.empty()

    def show_performance_metrics(self):
        metrics = self.task.metrics
        if not metrics:
            st.warning("No metrics available for this model.")
            return

        acc = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1_score"]
        st.write(f"Accuracy: {acc:.2%}")
        st.write(f"Precision: {precision:.2%}")
        st.write(f"Recall: {recall:.2%}")
        st.write(f"F1 Score: {f1:.2%}")

        st.write("Per-Class Metrics")
        df_report = pd.DataFrame(metrics["classification_report"]).T
        st.dataframe(df_report)

        st.write("Confusion Matrix")
        cm = pd.DataFrame(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            annot_kws={"size": 6}
        )
        ax.tick_params(axis='both', which='major', labelsize=3)
        ax.xaxis.label.set_size(3)
        ax.yaxis.label.set_size(3)
        st.pyplot(fig)

        if self.task.best_params:
            st.write("Best Parameters")
            st.write(self.task.best_params)

    def get_symptom_options(self) -> None:
        df = disease_symptoms_data.get_data()
        symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
        symptoms = []
        for col in symptom_cols:
            for val in df[col].dropna():
                if isinstance(val, str):
                    symptoms.append(val.strip())
                elif isinstance(val, list):
                    symptoms.extend([s.strip() for s in val if isinstance(s, str) and s.strip()])
        self.symptom_options = sorted(set(symptoms))
        self.symptom_map = {self.format_symptom_display(s): s for s in self.symptom_options}


    @staticmethod
    def format_symptom_display(symptom: str) -> str:
        return " ".join(word.capitalize() for word in symptom.split("_"))
