import streamlit as st
from client.page_modules import CancerPage, DiseasePage, PredictionPage
from config import TaskName

def start_client():
    task_choice = st.sidebar.radio("", [t.value for t in TaskName])

    if "page_modules" not in st.session_state:
        st.session_state.page_modules = {
            TaskName.CANCER_FEATURES.value: CancerPage(),
            TaskName.DISEASE_SYMPTOMS.value: DiseasePage(),
            TaskName.DISEASE_PREDICTION.value: PredictionPage(),
        }

    page = st.session_state.page_modules[task_choice]
    page.render()