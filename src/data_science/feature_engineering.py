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

import json
import os
import numpy as np
import pandas as pd


class FraudFeatureEngineer:
    """
    Feature Engineering Pipeline for Credit Card Fraud Detection
    """

    def __init__(
        self,
        target_col="is_fraud",
        drop_cols=None
    ):
        self.target_col = target_col

        self.drop_cols = drop_cols or [
            # IDs
            "trans_num", "cc_num",

            # personal info
            "first", "last", "street", "zip",

            # datetime raw
            "trans_date_trans_time", "dob",
            "unix_time", "prev_time",

            # location
            "lat", "long",
            "merch_lat", "merch_long",

            # raw categorical
            "merchant", "job", "city",

            # redundant engineered
            "amt_per_pop",
            "log_amt_per_pop",
            "amt_zscore",

            # duplicate aggregation
            "user_avg_amt"
        ]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform complete feature engineering pipeline
        """

        data = df.copy()

        # =========================================================
        # Amount Features
        # =========================================================

        data["amt_per_pop"] = (
            data["amt"] / (data["city_pop"] + 1e-6)
        )

        data["log_amt_per_pop"] = (
            np.log1p(data["amt"]) -
            np.log1p(data["city_pop"])
        )

        data["log_amt"] = np.log1p(data["amt"])

        data["amt_zscore"] = (
            (data["amt"] - data["amt"].mean()) /
            data["amt"].std()
        )

        # =========================================================
        # User Behaviour Features
        # =========================================================

        data["amt_user_mean"] = (
            data.groupby("cc_num")["amt"]
            .transform("mean")
        )

        data["amt_user_std"] = (
            data.groupby("cc_num")["amt"]
            .transform("std")
        )

        data["amt_user_zscore"] = (
            (data["amt"] - data["amt_user_mean"]) /
            (data["amt_user_std"] + 1e-6)
        )

        data["user_txn_count"] = (
            data.groupby("cc_num")["amt"]
            .transform("count")
        )

        data["user_avg_amt"] = (
            data.groupby("cc_num")["amt"]
            .transform("mean")
        )

        data["user_merchant_count"] = (
            data.groupby(["cc_num", "merchant"])["amt"]
            .transform("count")
        )

        # =========================================================
        # Datetime Features
        # =========================================================

        data["trans_date_trans_time"] = pd.to_datetime(
            data["trans_date_trans_time"],
            errors="coerce"
        )

        data["dob"] = pd.to_datetime(
            data["dob"],
            errors="coerce"
        )

        data["hour"] = (
            data["trans_date_trans_time"].dt.hour
        )

        data["day"] = (
            data["trans_date_trans_time"].dt.day
        )

        data["month"] = (
            data["trans_date_trans_time"].dt.month
        )

        data["weekday"] = (
            data["trans_date_trans_time"].dt.weekday
        )

        data["is_weekend"] = (
            data["weekday"].isin([5, 6]).astype(int)
        )

        data["age"] = (
            (
                data["trans_date_trans_time"] -
                data["dob"]
            ).dt.days // 365
        )

        # =========================================================
        # Time Difference Features
        # =========================================================

        data = data.sort_values(
            ["cc_num", "trans_date_trans_time"]
        )

        data["prev_time"] = (
            data.groupby("cc_num")
            ["trans_date_trans_time"]
            .shift(1)
        )

        data["time_diff"] = (
            data["trans_date_trans_time"] -
            data["prev_time"]
        ).dt.total_seconds()

        data["time_diff"] = (
            data["time_diff"]
            .fillna(data["time_diff"].median())
        )

        # =========================================================
        # Frequency Encoding
        # =========================================================

        merchant_freq = (
            data["merchant"].value_counts()
        )

        data["merchant_freq"] = (
            data["merchant"]
            .map(merchant_freq)
        )

        job_freq = (
            data["job"].value_counts()
        )

        data["job_freq"] = (
            data["job"]
            .map(job_freq)
        )

        city_freq = (
            data["city"].value_counts()
        )

        data["city_freq"] = (
            data["city"]
            .map(city_freq)
        )

        # =========================================================
        # Recent Transaction Features
        # =========================================================

        data["txn_last_1h"] = (
            data.groupby("cc_num")
            ["trans_date_trans_time"]
            .transform(
                lambda x: (
                    x.diff()
                    .dt.total_seconds()
                    .lt(3600)
                    .cumsum()
                )
            )
        )

        # =========================================================
        # One Hot Encoding
        # =========================================================

        data = pd.get_dummies(
            data,
            columns=["gender", "category", "state"],
            drop_first=True,
            dtype=int
        )

        # =========================================================
        # Drop Unwanted Columns
        # =========================================================

        data = data.drop(
            columns=self.drop_cols,
            errors="ignore"
        )

        return data

    def generate_metadata(
        self,
        data: pd.DataFrame
    ) -> dict:
        """
        Generate metadata dictionary
        """

        feature_cols = [
            col for col in data.columns
            if col != self.target_col
        ]

        num_cols = (
            data[feature_cols]
            .select_dtypes(include=["int64", "float64"])
            .columns
            .tolist()
        )

        cat_cols = (
            data[feature_cols]
            .select_dtypes(include=["object", "category"])
            .columns
            .tolist()
        )

        bool_cols = (
            data[feature_cols]
            .select_dtypes(include=["bool"])
            .columns
            .tolist()
        )

        engineered_features = {
            "amount_features": [
                col for col in feature_cols
                if "amt" in col
            ],

            "user_behavior_features": [
                col for col in feature_cols
                if "user" in col
            ],

            "time_features": [
                "hour",
                "day",
                "month",
                "weekday",
                "is_weekend",
                "time_diff",
                "txn_last_1h"
            ],

            "frequency_features": [
                col for col in feature_cols
                if "freq" in col
            ]
        }

        metadata = {
            "target_col": self.target_col,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "bool_cols": bool_cols,
            "engineered_features": engineered_features,
            "total_features": len(feature_cols)
        }

        return metadata

    def save_metadata(
        self,
        metadata: dict,
        save_path: str
    ):
        """
        Save metadata as JSON
        """

        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def save_data(
        self,
        data: pd.DataFrame,
        save_path: str
    ):
        """
        Save processed dataframe
        """

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        data.to_csv(save_path, index=False)

        print(f"Data saved at: {save_path}")
