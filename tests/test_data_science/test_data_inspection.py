import pandas as pd
import pytest

# import your module here
# from your_module import (
#     inspect_shape,
#     inspect_types,
#     preview_data,
#     preview_tail,
#     missing_values_summary,
#     unique_values_summary,
#     basic_info
# )

from your_module import (
    inspect_shape,
    inspect_types,
    preview_data,
    preview_tail,
    missing_values_summary,
    unique_values_summary,
    basic_info
)


@pytest.fixture
def sample_df():
    data = {
        "A": [1, 2, None, 4],
        "B": ["x", "y", "y", None],
        "C": [10.5, 20.1, 30.2, 40.3]
    }
    return pd.DataFrame(data)


def test_inspect_shape(sample_df):
    assert inspect_shape(sample_df) == (4, 3)


def test_inspect_types(sample_df):
    dtypes = inspect_types(sample_df)
    assert dtypes["A"] == "float64"
    assert dtypes["B"] == "object"
    assert dtypes["C"] == "float64"


def test_preview_data(sample_df):
    result = preview_data(sample_df, n=2)
    assert len(result) == 2
    assert result.iloc[0]["A"] == 1


def test_preview_tail(sample_df):
    result = preview_tail(sample_df, n=2)
    assert len(result) == 2
    assert result.iloc[-1]["A"] == 4


def test_missing_values_summary(sample_df):
    result = missing_values_summary(sample_df).to_dict()
    assert result["A"] == 1
    assert result["B"] == 1
    assert result["C"] == 0


def test_unique_values_summary(sample_df):
    result = unique_values_summary(sample_df)
    assert result["A"] == 3  # [1,2,4]
    assert result["B"] == 2   # ["x","y"]
    assert result["C"] == 4


def test_basic_info(sample_df):
    result = basic_info(sample_df)

    assert result["shape"] == (4, 3)

    assert result["missing_values"]["A"] == 1
    assert result["missing_values"]["B"] == 1

    assert result["unique_values"]["B"] == 2
    assert "A" in result["dtypes"]