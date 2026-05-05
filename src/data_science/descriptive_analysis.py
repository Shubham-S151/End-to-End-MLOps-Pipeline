import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary statistics for numerical features.
    Includes mean, std, min, max, and percentiles.
    """
    return df.describe()

def plot_histograms(df: pd.DataFrame, num_cols: list, bins: int = 30):
    """
    Plot histograms for numerical columns.
    """
    figs = {}
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col], bins=bins, kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        figs[col] = fig
    return figs

def plot_boxplots(df: pd.DataFrame, num_cols: list):
    """
    Plot boxplots for numerical columns.
    """
    figs = {}
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        figs[col] = fig
    return figs

def plot_countplots(df: pd.DataFrame, cat_cols: list):
    """
    Plot count plots for categorical columns.
    """
    figs = {}
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=df[col], ax=ax)
        ax.set_title(f"Count Plot of {col}")
        figs[col] = fig
    return figs

def correlation_heatmap(df: pd.DataFrame):
    """
    Plot correlation heatmap for numerical features.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

def pairwise_scatter(df: pd.DataFrame, cols: list):
    """
    Plot pairwise scatter plots for selected columns.
    """
    fig = sns.pairplot(df[cols])
    plt.suptitle("Pairwise Scatter Plots", y=1.02)
    return fig
