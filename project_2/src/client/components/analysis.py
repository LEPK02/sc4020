from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


class AnalysisBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._title: Optional[str] = None
        self._x: Optional[str] = None
        self._y: Optional[str] = None
        self._hue: Optional[str] = None
        self._kind: str = "bar"
        self._aggfunc: Optional[Callable] = None
        self._top_n: Optional[int] = None

    # --- Builder pattern methods ---
    def title(self, title: str) -> "AnalysisBuilder":
        self._title = title
        return self

    def x(self, column: str) -> "AnalysisBuilder":
        self._x = column
        return self

    def y(self, column: str) -> "AnalysisBuilder":
        self._y = column
        return self

    def hue(self, column: str) -> "AnalysisBuilder":
        self._hue = column
        return self

    def kind(self, chart_type: str) -> "AnalysisBuilder":
        self._kind = chart_type
        return self

    def aggregate(self, func: Callable) -> "AnalysisBuilder":
        self._aggfunc = func
        return self

    def top(self, n: int) -> "AnalysisBuilder":
        self._top_n = n
        return self

    # --- Render graph ---
    def render(self):
        if not self._x:
            st.warning("Select an X-axis column to plot.")
            return

        df = self.df.copy()

        # Optional aggregation
        if self._aggfunc and self._y:
            df = df.groupby(self._x)[self._y].agg(self._aggfunc).reset_index()

        # Optional top N filtering
        if self._top_n and self._x in df.columns:
            df = (
                df.nlargest(self._top_n, self._y)
                if self._y in df.columns
                else df.head(self._top_n)
            )

        # --- Plotting logic ---
        plt.figure(figsize=(8, 5))
        if self._kind == "bar":
            sns.barplot(data=df, x=self._x, y=self._y, hue=self._hue)
        elif self._kind == "line":
            sns.lineplot(data=df, x=self._x, y=self._y, hue=self._hue, marker="o")
        elif self._kind == "hist":
            sns.histplot(data=df, x=self._x, hue=self._hue, bins=20, kde=True)
        elif self._kind == "box":
            sns.boxplot(data=df, x=self._x, y=self._y, hue=self._hue)
        elif self._kind == "scatter":
            sns.scatterplot(data=df, x=self._x, y=self._y, hue=self._hue)
        else:
            st.error(f"Unsupported chart type: {self._kind}")
            return

        plt.title(self._title or "Data Analysis")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()


class Analysis:
    """Convenience wrapper for common dataset analyses."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary(self):
        st.write("### Dataset Overview")
        st.write(self.df.describe(include="all"))

    def correlation(self):
        st.write("### Correlation Heatmap")
        numeric_df = self.df.select_dtypes(include=["number"])
        if numeric_df.empty:
            st.info("No numeric data to plot correlation heatmap.")
            return
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
        plt.close()

    def categorical_distribution(self, column: str):
        st.write(f"### Distribution of {column}")
        AnalysisBuilder(self.df).title(f"Distribution of {column}").x(column).kind(
            "bar"
        ).render()

    def feature_vs_target(self, feature: str, target: str):
        st.write(f"### {feature} vs {target}")
        AnalysisBuilder(self.df).title(f"{feature} vs {target}").x(feature).y(
            target
        ).kind("box").render()

    # TODO: make data analysis table and graphs
    # Then call in respective file under page_modules > cancer.py / disease.py > CancerPage.render_analysis()
