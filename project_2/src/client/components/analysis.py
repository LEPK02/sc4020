from typing import List, Dict
import pandas as pd
import numpy as np
import streamlit as st

class Analysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.strategies: List[str] = sorted(df["strategy"].unique().tolist())
        self.top_n: int = 5
        self.ranks: List[int] = list(range(1, self.top_n + 1))

    def _table(self, data: Dict[str, List[str]]) -> pd.DataFrame:
        df = pd.DataFrame({"Rank": self.ranks}).set_index("Rank")
        for k, v in data.items():
            df[k] = v
        return df

    def render_support(self) -> "Analysis":
        table_m: Dict[str, List[str]] = {}
        table_b: Dict[str, List[str]] = {}

        for strat in self.strategies:
            sdf = self.df[self.df["strategy"] == strat]
            m_sorted = sdf.sort_values("support_percent_m", ascending=False)
            b_sorted = sdf.sort_values("support_percent_b", ascending=False)
            table_m[strat] = m_sorted["pattern"].head(self.top_n).tolist()
            table_b[strat] = b_sorted["pattern"].head(self.top_n).tolist()

        st.subheader("Top Patterns by Support Percentage")
        st.write("Malignant")
        st.dataframe(self._table(table_m), width='stretch')
        st.write("Benign")
        st.dataframe(self._table(table_b), width='stretch')
        return self

    def render_contrast(self) -> "Analysis":
        table_m: Dict[str, List[str]] = {}
        table_b: Dict[str, List[str]] = {}

        for strat in self.strategies:
            sdf = self.df[self.df["strategy"] == strat]
            m_sorted = sdf.sort_values(["Contrast_Rate", "support_percent_m"], ascending=[False, False])
            b_sorted = sdf.sort_values(["Contrast_Rate", "support_percent_b"], ascending=[True, False])
            table_m[strat] = m_sorted["pattern"].head(self.top_n).tolist()
            table_b[strat] = b_sorted["pattern"].head(self.top_n).tolist()

        st.subheader("Top Patterns by Contrast Rate")
        st.write("Malignant-Leaning")
        st.dataframe(self._table(table_m), width='stretch')
        st.write("Benign-Leaning")
        st.dataframe(self._table(table_b), width='stretch')
        return self

    def render_specific(self) -> "Analysis":
        table_m: Dict[str, List[str]] = {}
        table_b: Dict[str, List[str]] = {}

        exclusive = self.df[self.df["Contrast_Rate"] == np.inf]

        for strat in self.strategies:
            sdf_m = exclusive[(exclusive["strategy"] == strat) & (exclusive["support_percent_m"] > 0)]
            sdf_b = exclusive[(exclusive["strategy"] == strat) & (exclusive["support_percent_b"] > 0)]

            m_sorted = sdf_m.sort_values(["pattern_length", "support_percent_m"], ascending=[False, False])
            b_sorted = sdf_b.sort_values(["pattern_length", "support_percent_b"], ascending=[False, False])

            m_list = m_sorted["pattern"].head(self.top_n).tolist()
            b_list = b_sorted["pattern"].head(self.top_n).tolist()

            if len(m_list) < self.top_n:
                m_list += ["-"] * (self.top_n - len(m_list))
            if len(b_list) < self.top_n:
                b_list += ["-"] * (self.top_n - len(b_list))

            table_m[strat] = m_list
            table_b[strat] = b_list

        st.subheader("Top Exclusive (Specific) Patterns")
        st.write("Exclusive Malignant")
        st.dataframe(self._table(table_m), width='stretch')
        st.write("Exclusive Benign")
        st.dataframe(self._table(table_b), width='stretch')
        return self
