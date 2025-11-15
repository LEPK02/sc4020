
from algo import GSPAlgo
from client.components import Analysis, Filters
from config import TaskName
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
        Analysis(self.df).render_support().render_contrast().render_specific()
