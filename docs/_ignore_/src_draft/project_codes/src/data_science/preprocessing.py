import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def impute_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values by imputation.
    Supported strategies: 'mean', 'median', 'mode'.
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def normalize_features(df: pd.DataFrame, cols: list, method: str = "standard") -> pd.DataFrame:
    """
    Normalize or standardize numerical features.
    method: 'standard' (z-score) or 'minmax' (0–1 scaling).
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def transform_categorical(df: pd.DataFrame, cat_cols: list, method: str = "onehot") -> pd.DataFrame:
    """
    Transform categorical variables.
    method: 'onehot' or 'label'.
    """
    if method == "onehot":
        return pd.get_dummies(df, columns=cat_cols, drop_first=True)
    elif method == "label":
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df
    else:
        raise ValueError("Unsupported method. Use 'onehot' or 'label'.")

def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """
    Split data into train, validation, and test sets.
    """
    X = df.drop(columns=[target])
    y = df[target]

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction, random_state=random_state, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
