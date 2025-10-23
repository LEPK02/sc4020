from algo import GSPAlgo
from config import TaskName
from components import Analysis, Filters
from data import cancer_data

from .base import BasePage


class CancerPage(BasePage):
    def __init__(self):
        super().__init__(
            task_loader=lambda: GSPAlgo(cancer_data),
            title=f"{TaskName.CANCER_FEATURES.value} Analysis",
            display_title="Patient Feature Sequences (GSP)"
        )

    def render_filters(self):
        filters = Filters(self.df)
        self.df = filters.render_filters()

    def render_analysis(self):
        analysis = Analysis(self.df)
        analysis.summary()
        analysis.correlation()
        if "diagnosis" in self.df.columns:
            analysis.feature_vs_target("feature_1_value", "diagnosis")

