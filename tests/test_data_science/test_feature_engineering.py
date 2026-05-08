import pandas as pd
import numpy as np
import pytest

from src.data_science.feature_engineering import (
    create_date_features,
    encode_label,
    encode_onehot,
    scale_features,
    select_features,
)


def test_create_date_features():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-06"]
    })

    result = create_date_features(df, "date")

    assert "date_year" in result.columns
    assert "date_month" in result.columns
    assert "date_day" in result.columns
    assert "date_dayofweek" in result.columns
    assert "date_is_weekend" in result.columns

    assert result.loc[0, "date_year"] == 2024
    assert result.loc[0, "date_month"] == 1
    assert result.loc[0, "date_day"] == 1

    # 2024-01-01 is Monday -> 0
    assert result.loc[0, "date_dayofweek"] == 0

    # 2024-01-06 is Saturday -> weekend
    assert result.loc[1, "date_is_weekend"] == 1


def test_encode_label():
    df = pd.DataFrame({
        "color": ["red", "blue", "red", "green"]
    })

    result = encode_label(df, "color")

    assert result["color"].dtype in [np.int32, np.int64]

    # Ensure unique labels encoded
    assert len(result["color"].unique()) == 3


def test_encode_onehot():
    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai", "Delhi"],
        "value": [1, 2, 3]
    })

    result = encode_onehot(df, ["city"])

    # drop_first=True removes one category
    assert "city_Mumbai" in result.columns or "city_Delhi" in result.columns

    # Original column should be removed
    assert "city" not in result.columns


def test_scale_features_standard():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5]
    })

    result = scale_features(df, ["a"], method="standard")

    # Mean should be approximately 0
    assert pytest.approx(result["a"].mean(), abs=1e-6) == 0


def test_scale_features_minmax():
    df = pd.DataFrame({
        "a": [10, 20, 30]
    })

    result = scale_features(df, ["a"], method="minmax")

    assert result["a"].min() == 0
    assert result["a"].max() == 1


def test_select_features():
    np.random.seed(42)

    df = pd.DataFrame({
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "f3": np.random.rand(100),
        "target": np.random.randint(0, 2, 100)
    })

    result = select_features(
        df=df,
        target="target",
        num_cols=["f1", "f2", "f3"],
        k=2
    )

    assert isinstance(result, pd.Series)

    # Should return top 2 features
    assert len(result) == 2

    # Feature names should belong to input columns
    assert all(col in ["f1", "f2", "f3"] for col in result.index)