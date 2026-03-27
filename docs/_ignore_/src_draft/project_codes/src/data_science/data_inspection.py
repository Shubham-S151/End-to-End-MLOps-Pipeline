import pandas as pd

def inspect_shape(df: pd.DataFrame) -> tuple:
    """
    Return the shape of the dataframe (rows, columns).
    """
    return df.shape

def inspect_types(df: pd.DataFrame) -> pd.Series:
    """
    Return the data types of each column.
    """
    return df.dtypes

def preview_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the first n rows of the dataframe.
    """
    return df.head(n)

def preview_tail(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the last n rows of the dataframe.
    """
    return df.tail(n)

def missing_values_summary(df: pd.DataFrame) -> pd.Series:
    """
    Return the count of missing values per column.
    """
    return df.isnull().sum()

def unique_values_summary(df: pd.DataFrame) -> dict:
    """
    Return the number of unique values per column.
    """
    return {col: df[col].nunique() for col in df.columns}

def basic_info(df: pd.DataFrame) -> dict:
    """
    Return a dictionary with basic dataset information:
    shape, dtypes, missing values, unique counts.
    """
    return {
        "shape": inspect_shape(df),
        "dtypes": inspect_types(df).to_dict(),
        "missing_values": missing_values_summary(df).to_dict(),
        "unique_values": unique_values_summary(df)
    }
