import pandas as pd
import numpy as np
import pytest

from data_cleaning import (
    check_missing,
    check_duplicates,
    remove_duplicates,
    detect_outliers_iqr,
    remove_outliers_iqr,
    fix_inconsistencies,
    convert_dtypes,
)

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, 2, 3, np.nan],
        "B": [" X ", "y", "y", "Z", "z"],
        "C": [10, 20, 20, 30, 1000],  # outlier: 1000
    })


# -------------------------
# Missing values
# -------------------------
def test_check_missing(sample_df):
    result = check_missing(sample_df)
    assert result["A"] == 1
    assert result["B"] == 0
    assert result["C"] == 0


# -------------------------
# Duplicates
# -------------------------
def test_check_duplicates(sample_df):
    assert check_duplicates(sample_df) == 1


def test_remove_duplicates(sample_df):
    df = remove_duplicates(sample_df)
    assert len(df) == len(sample_df) - 1
    assert df.duplicated().sum() == 0


# -------------------------
# Outliers (IQR)
# -------------------------
def test_detect_outliers_iqr(sample_df):
    outliers = detect_outliers_iqr(sample_df, "C")
    assert (outliers["C"] == 1000).any()
    assert len(outliers) == 1


def test_remove_outliers_iqr(sample_df):
    cleaned = remove_outliers_iqr(sample_df, "C")
    assert 1000 not in cleaned["C"].values
    assert cleaned["C"].max() < 1000


# -------------------------
# Fix inconsistencies
# -------------------------
def test_fix_inconsistencies(sample_df):
    df = sample_df.copy()
    fixed = fix_inconsistencies(df, "B")

    assert fixed["B"].iloc[0] == "x"
    assert fixed["B"].iloc[1] == "y"
    assert fixed["B"].iloc[3] == "z"


# -------------------------
# Type conversion
# -------------------------
def test_convert_dtypes():
    df = pd.DataFrame({
        "A": ["1", "2", "3"],
        "B": ["1.1", "2.2", "3.3"]
    })

    converted = convert_dtypes(df, {"A": "int", "B": "float"})

    assert converted["A"].dtype == "int64"
    assert converted["B"].dtype == "float64"
    assert converted["A"].sum() == 6
    assert np.isclose(converted["B"].sum(), 6.6)
# 