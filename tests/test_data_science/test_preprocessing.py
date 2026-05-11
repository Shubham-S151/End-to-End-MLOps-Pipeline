import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import pytest

from src.data_science.preprocessing import (
    impute_missing,
    normalize_features,
    transform_categorical,
    split_data,
)


# -----------------------------
# Tests for impute_missing
# -----------------------------

def test_impute_missing_mean():
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": [10, 20, 30, 40]
    })

    result = impute_missing(df.copy(), strategy="mean")

    expected_mean = (1 + 2 + 4) / 3
    assert result["A"].isnull().sum() == 0
    assert result.loc[2, "A"] == expected_mean


def test_impute_missing_median():
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 100]
    })

    result = impute_missing(df.copy(), strategy="median")

    expected_median = 2
    assert result.loc[2, "A"] == expected_median


def test_impute_missing_mode():
    df = pd.DataFrame({
        "A": ["x", "y", None, "x"]
    })

    result = impute_missing(df.copy(), strategy="mode")

    assert result.loc[2, "A"] == "x"


# -----------------------------
# Tests for normalize_features
# -----------------------------

def test_normalize_features_standard():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5]
    })

    result = normalize_features(df.copy(), ["A"], method="standard")

    # Mean should be close to 0
    assert pytest.approx(result["A"].mean(), abs=1e-7) == 0

    # Std should be close to 1
    assert pytest.approx(result["A"].std(ddof=0), abs=1e-7) == 1


def test_normalize_features_minmax():
    df = pd.DataFrame({
        "A": [10, 20, 30]
    })

    result = normalize_features(df.copy(), ["A"], method="minmax")

    assert result["A"].min() == 0
    assert result["A"].max() == 1


# -----------------------------
# Tests for transform_categorical
# -----------------------------

def test_transform_categorical_onehot():
    df = pd.DataFrame({
        "color": ["red", "blue", "red"]
    })

    result = transform_categorical(df.copy(), ["color"], method="onehot")

    # drop_first=True removes one category
    assert "color_red" in result.columns or "color_blue" in result.columns
    assert "color" not in result.columns


def test_transform_categorical_label():
    df = pd.DataFrame({
        "color": ["red", "blue", "green"]
    })

    result = transform_categorical(df.copy(), ["color"], method="label")

    assert pd.api.types.is_integer_dtype(result["color"])
    assert len(result["color"].unique()) == 3


def test_transform_categorical_invalid_method():
    df = pd.DataFrame({
        "color": ["red", "blue"]
    })

    with pytest.raises(ValueError):
        transform_categorical(df, ["color"], method="invalid")


# -----------------------------
# Tests for split_data
# -----------------------------

def test_split_data_shapes():
    df = pd.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": [0, 1] * 50
    })

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target="target",
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # Total rows preserved
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(df)

    # Check approximate split sizes
    assert len(X_test) == 20
    assert len(X_val) == 10
    assert len(X_train) == 70

    # Ensure target removed from features
    assert "target" not in X_train.columns


def test_split_data_stratification():
    df = pd.DataFrame({
        "feature": range(100),
        "target": [0] * 50 + [1] * 50
    })

    _, _, _, y_train, y_val, y_test = split_data(df, target="target")

    # Stratification should preserve class balance
    assert pytest.approx(y_train.mean(), abs=0.1) == 0.5
    assert pytest.approx(y_val.mean(), abs=0.1) == 0.5
    assert pytest.approx(y_test.mean(), abs=0.1) == 0.5
