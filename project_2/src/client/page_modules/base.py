from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd
import streamlit as st
from algo import BaseAlgo


class BasePage(ABC):
    def __init__(
        self, task_loader: Callable[[], BaseAlgo], title: str, display_title: str
    ):
        self.title = title
        self.display_title = display_title
        with st.spinner("Loading data…"):
            self.task: BaseAlgo = task_loader()
            self.df: pd.DataFrame = self.task.run()

    def get_task(self) -> BaseAlgo:
        return self.task

    def get_df(self) -> pd.DataFrame:
        return self.df.round(2)
    
    @abstractmethod
    def render_filters(self):
        ...

    def render_data_editor(self):
        self.render_filters()
        st.data_editor(self.df)

    @abstractmethod
    def render_analysis(self):
        ...

    def render(self):
        st.title(self.title)
        st.subheader(self.display_title)
        self.render_data_editor()
        self.render_analysis()
