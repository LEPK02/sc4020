import streamlit as st
from page_modules import CancerPage, DiseasePage
from config import TaskName

task_choice = st.sidebar.radio("", [t.value for t in TaskName])

if "page_modules" not in st.session_state:
    st.session_state.page_modules = {
        TaskName.CANCER_FEATURES.value: CancerPage(),
        TaskName.DISEASE_SYMPTOMS.value: DiseasePage(),
    }

page = st.session_state.page_modules[task_choice]
page.render()
