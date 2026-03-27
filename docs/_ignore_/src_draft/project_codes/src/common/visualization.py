import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Univariate Analysis
class UnivariateAnalysis:
    def __init__(self, data: pd.DataFrame, columns: dict):
        self.df = data.copy()
        self.target_col = columns.get("target_col", None)
        self.num_cols = columns.get("num_cols", [])
        self.cat_cols = columns.get("cat_cols", [])
        self.time_cols = columns.get("time_cols", [])
        self.uniq_cols = columns.get("uniq_cols", [])
        self.loc_cols = columns.get("loc_cols", [])

    def density_plot(self, col: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(self.df[col], fill=True, ax=ax)
        ax.set_title(f"Density Plot for {col}")
        return fig

    def box_plot(self, col: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=self.df[col], ax=ax)
        ax.set_title(f"Box Plot for {col}")
        return fig

    def distribution_plot(self, col: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.df[col], bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution Plot for {col}")
        return fig

    def pie_plot(self, col: str):
        counts = self.df[col].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Pie Chart for {col}")
        return fig

    def plot(self):
        figs = {}
        for col in self.num_cols:
            figs[f"{col}_density"] = self.density_plot(col)
            figs[f"{col}_box"] = self.box_plot(col)
            figs[f"{col}_distribution"] = self.distribution_plot(col)
        for col in self.cat_cols:
            figs[f"{col}_pie"] = self.pie_plot(col)
        return figs

# Bivariate Analysis
class BivariateAnalysis:
    def __init__(self, data: pd.DataFrame, columns: dict):
        self.df = data.copy()
        self.target_col = columns.get("target_col", None)
        self.num_cols = columns.get("num_cols", [])
        self.cat_cols = columns.get("cat_cols", [])

    def scatter_plot(self, x: str, y: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=self.df[x], y=self.df[y], ax=ax)
        ax.set_title(f"Scatter Plot: {x} vs {y}")
        return fig

    def correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = self.df[self.num_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        return fig

    def grouped_boxplot(self, num_col: str, cat_col: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=self.df[cat_col], y=self.df[num_col], ax=ax)
        ax.set_title(f"Boxplot of {num_col} grouped by {cat_col}")
        return fig

# Time Series Analysis
class TimeSeriesAnalysis:
    def __init__(self, data: pd.DataFrame, columns: dict):
        self.df = data.copy()
        self.time_cols = columns.get("time_cols", [])
        self.num_cols = columns.get("num_cols", [])

    def line_plot(self, col: str, time_col: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=self.df[time_col], y=self.df[col], ax=ax)
        ax.set_title(f"Time Series Plot: {col} over {time_col}")
        return fig

    def rolling_average(self, col: str, window: int = 7):
        fig, ax = plt.subplots(figsize=(10, 5))
        self.df[col].rolling(window).mean().plot(ax=ax)
        ax.set_title(f"Rolling Average ({window}) of {col}")
        return fig

    def seasonal_decompose(self, col: str, time_col: str, period: int = 12):
        ts = self.df.set_index(time_col)[col]
        decomposition = seasonal_decompose(ts, period=period)
        fig = decomposition.plot()
        return fig

# Custom Plots
class CustomPlots:
    def __init__(self, data: pd.DataFrame, columns: dict):
        self.df = data.copy()
        self.num_cols = columns.get("num_cols", [])
        self.target_col = columns.get("target_col", None)

    def correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = self.df[self.num_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        return fig

    def pairplot(self, cols: list = None):
        cols = cols or self.num_cols
        fig = sns.pairplot(self.df[cols])
        return fig

    def target_vs_feature_plot(self, feature: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=self.df[self.target_col], y=self.df[feature], ax=ax)
        ax.set_title(f"{feature} vs Target ({self.target_col})")
        return fig
