from typing import Any
import pandas as pd
import streamlit as st


class Filters:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.filters: dict[str, Any] = {}

    def render_filters(self) -> pd.DataFrame:
        filtered_df = self.df.copy()

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = float(self.df[col].min())
                max_val = float(self.df[col].max())

                if min_val == max_val:
                    st.sidebar.write(f"{col}: {min_val} (all values identical)")
                    self.filters[col] = (min_val, max_val)
                else:
                    selected = st.sidebar.slider(
                        f"Filter {col}", min_val, max_val, (min_val, max_val)
                    )
                    self.filters[col] = selected
                    filtered_df = filtered_df[
                        (filtered_df[col] >= selected[0]) & (filtered_df[col] <= selected[1])
                    ]
            else:
                options = sorted(self.df[col].dropna().unique().tolist())
                selected = st.sidebar.multiselect(f"Filter {col}", options, default=options)
                self.filters[col] = selected
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

        return filtered_df
