from algo import AprioriAlgo
from config import TaskName
from components import Filters
from data import disease_symptoms_data

from .base import BasePage


class DiseasePage(BasePage):
    def __init__(self):
        super().__init__(
            task_loader=lambda: AprioriAlgo(disease_symptoms_data),
            title=f"{TaskName.DISEASE_SYMPTOMS.value} Analysis",
            display_title="Frequent Symptom Patterns (Apriori)"
        )

    def render_filters(self):
        filters = Filters(self.df)
        self.df = filters.render_filters()

    def render_analysis(self):
        pass