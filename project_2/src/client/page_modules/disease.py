from typing import Set, Tuple, cast

import pandas as pd
import streamlit as st
from algo import AprioriAlgo
from client.components import Filters
from config import TaskName
from data import disease_symptoms_data

from .base import BasePage


class DiseasePage(BasePage):
    def __init__(self):
        super().__init__(
            task_loader=lambda: AprioriAlgo(disease_symptoms_data),
            title=f"{TaskName.DISEASE_SYMPTOMS.value} Analysis",
            display_title="Frequent Symptom Patterns (Apriori)",
        )

    def render_filters(self):
        filters = Filters(self.df)
        self.df = filters.render_filters()

    def render_analysis(self):
        task = cast(AprioriAlgo, self.get_task())
        disease_map = task.find_diseases_for_symptom_pairs(
            disease_symptoms_data.get_data(),
            cast(
                Set[Tuple[str, str]],
                {
                    tuple(sorted([str(a).strip(), str(b).strip()]))
                    for a, b in zip(
                        self.get_df()["antecedent"], self.get_df()["consequent"]
                    )
                },
            ),
        )

        st.subheader("🧬 Diseases Associated with Symptom Pairs")

        if not disease_map:
            st.info("No disease associations found for the discovered symptom pairs.")
            return

        display_data = []
        for pair, diseases in disease_map.items():
            bullet_list = "<br>".join(f"• {d}" for d in sorted(diseases))
            display_data.append({"Symptom Pair": pair, "Diseases": bullet_list})

        df_display = pd.DataFrame(display_data)
        
        html_table = df_display.to_html(
            escape=False,
            index=False,
        )
        html_table = html_table.replace(
            "<th>", '<th style="text-align: left;">'
        )

        st.write(html_table, unsafe_allow_html=True)
