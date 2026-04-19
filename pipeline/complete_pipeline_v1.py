import numpy as np
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# 1. Custom Feature Engineer
class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Basic numeric features
        X["log_amt"] = np.log1p(X["amt"])
        X["amt_per_pop"] = X["amt"] / (X["city_pop"] + 1e-6)

        # Datetime conversion
        X["trans_date_trans_time"] = pd.to_datetime(X["trans_date_trans_time"])
        X["dob"] = pd.to_datetime(X["dob"])

        # Time features
        X["hour"] = X["trans_date_trans_time"].dt.hour
        X["day"] = X["trans_date_trans_time"].dt.day
        X["month"] = X["trans_date_trans_time"].dt.month
        X["weekday"] = X["trans_date_trans_time"].dt.weekday
        X["is_weekend"] = X["weekday"].isin([5, 6]).astype(int)

        # Age
        X["age"] = (X["trans_date_trans_time"] - X["dob"]).dt.days // 365

        return X


# 2. Column definitions
NUMERIC_FEATURES = [
    "amt", "city_pop", "log_amt", "amt_per_pop",
    "hour", "day", "month", "weekday", "is_weekend", "age"
]

CATEGORICAL_FEATURES = [
    "gender", "category", "state"
]


# 3. Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, NUMERIC_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
])


# 4. Full pipeline builder
def build_pipeline(model):
    pipe = Pipeline([
        ("feature_engineering", FraudFeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    return pipe


# 5. Save pipeline
def save_pipeline(pipe, path="pipeline.pkl"):
    with open(path, "wb") as f:
        pickle.dump(pipe, f)


# 6. Load pipeline
def load_pipeline(path="pipeline.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)