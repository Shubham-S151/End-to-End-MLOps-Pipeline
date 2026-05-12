from src.common import data_validation

import pytest
import pandas as pd
from src.common.data_validation import (
    validate_schema,
    check_missing_values,
    check_duplicates,
    check_unique,
    validate_value_ranges,
    validate_categories
)

# -------------------------
# Fixtures (sample data) 
# -------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 2, 4],
        "age": [25, 30, None, 40],
        "category": ["A", "B", "B", "C"]
    })


# -------------------------
# validate_schema tests
# -------------------------
def test_validate_schema_success(sample_df):
    schema = {"id": "int64", "age": "float64", "category": "object"}
    assert validate_schema(sample_df, schema) is True


def test_validate_schema_missing_column(sample_df):
    schema = {"id": "int64", "missing_col": "int64"}
    with pytest.raises(ValueError):
        validate_schema(sample_df, schema)


def test_validate_schema_wrong_dtype(sample_df):
    schema = {"id": "object"}  # wrong expected type
    with pytest.raises(TypeError):
        validate_schema(sample_df, schema)


# -------------------------
# check_missing_values tests
# -------------------------
def test_check_missing_values(sample_df):
    result = check_missing_values(sample_df)
    assert round(result["age"], 2) == 0.25  # 1 missing out of 4 rows


# -------------------------
# check_duplicates tests
# -------------------------
def test_check_duplicates(sample_df):
    assert check_duplicates(sample_df) == 0


# -------------------------
# check_unique tests
# -------------------------
def test_check_unique(sample_df):
    result = check_unique(sample_df, ["id", "category"])
    assert result["id"] is False   # duplicates exist
    assert result["category"] is False


# -------------------------
# validate_value_ranges tests
# -------------------------
def test_validate_value_ranges(sample_df):
    ranges = {"age": (20, 35)}
    result = validate_value_ranges(sample_df, ranges)
    assert result["age"] == 1  # one invalid value (40)


# -------------------------
# validate_categories tests
# -------------------------
def test_validate_categories_success(sample_df):
    assert validate_categories(sample_df, "category", ["A", "B", "C"]) is True


def test_validate_categories_failure(sample_df):
    with pytest.raises(ValueError):
        validate_categories(sample_df, "category", ["A", "B"])  # "C" invalid