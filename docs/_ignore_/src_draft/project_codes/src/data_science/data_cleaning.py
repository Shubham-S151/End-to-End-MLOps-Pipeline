import pandas as pd
import numpy as np

def check_missing(df: pd.DataFrame) -> pd.Series:
    """
    Return the count of missing values per column.
    """
    return df.isnull().sum()

def check_duplicates(df: pd.DataFrame) -> int:
    """
    Return the number of duplicate rows in the dataframe.
    """
    return df.duplicated().sum()

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.
    """
    return df.drop_duplicates()

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Detect outliers in a numerical column using the IQR method.
    Returns rows considered outliers.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
    return df[mask]

def remove_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Remove outliers from a numerical column using the IQR method.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    return df[mask]

def fix_inconsistencies(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Fix text formatting inconsistencies in a categorical column.
    Example: strip whitespace, convert to lowercase.
    """
    df[col] = df[col].astype(str).str.strip().str.lower()
    return df

def convert_dtypes(df: pd.DataFrame, conversions: dict) -> pd.DataFrame:
    """
    Convert data types of specified columns.
    conversions: dict like {"col1": "int", "col2": "float"}
    """
    for col, dtype in conversions.items():
        df[col] = df[col].astype(dtype)
    return df
