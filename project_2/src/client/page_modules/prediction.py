import time
import streamlit as st
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
            if self.task.accuracy is not None:
                st.write(f"Model accuracy: {self.task.accuracy:.2%}")

            st.write("5 diseases you are most likely to have:")

            top_results_display = top_results.reset_index()
            top_results_display.columns = ["Disease Name", "Probability of Disease"]
            st.dataframe(
                top_results_display.style.format({"Probability of Disease": "{:.2%}"}),
                width='stretch',
                hide_index=True,
            )

            status_placeholder = st.empty()
            status_placeholder.success("Diagnosis complete!")
            time.sleep(3)
            status_placeholder.empty()

    def get_symptom_options(self) -> None:
        df = disease_symptoms_data.get_data()
        symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
        self.symptom_options = sorted(
            {
                s.strip()
                for s in df[symptom_cols].stack().dropna()
                if isinstance(s, str) and s.strip()
            }
        )
        self.symptom_map = {self.format_symptom_display(s): s for s in self.symptom_options}

    @staticmethod
    def format_symptom_display(symptom: str) -> str:
        return " ".join(word.capitalize() for word in symptom.split("_"))
