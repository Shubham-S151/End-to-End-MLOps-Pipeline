import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

def create_date_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Create new date-based features from a datetime column.
    Features: year, month, day, day_of_week, is_weekend.
    """
    df[time_col] = pd.to_datetime(df[time_col])
    df[f"{time_col}_year"] = df[time_col].dt.year
    df[f"{time_col}_month"] = df[time_col].dt.month
    df[f"{time_col}_day"] = df[time_col].dt.day
    df[f"{time_col}_dayofweek"] = df[time_col].dt.dayofweek
    df[f"{time_col}_is_weekend"] = df[time_col].dt.dayofweek.isin([5, 6]).astype(int)
    return df

def encode_label(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Apply label encoding to a categorical column.
    """
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    return df

def encode_onehot(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns.
    """
    return pd.get_dummies(df, columns=cols, drop_first=True)

def scale_features(df: pd.DataFrame, cols: list, method: str = "standard") -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler or MinMaxScaler.
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def select_features(df: pd.DataFrame, target: str, num_cols: list, k: int = 10) -> pd.Series:
    """
    Select top k features based on mutual information with the target.
    Returns a Series of feature scores.
    """
    X = df[num_cols].fillna(0)
    y = df[target]
    scores = mutual_info_classif(X, y, discrete_features=False)
    return pd.Series(scores, index=num_cols).sort_values(ascending=False).head(k)
