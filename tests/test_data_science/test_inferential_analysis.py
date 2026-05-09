import pytest
import pandas as pd
import numpy as np

from src.data_science.inferential_analysis import (
    t_test,
    anova_test,
    chi_squared_test,
    correlation_test,
    normality_check,
)


def test_t_test_returns_expected_keys():
    group1 = pd.Series([1, 2, 3, 4, 5])
    group2 = pd.Series([2, 3, 4, 5, 6])

    result = t_test(group1, group2)

    assert "t_statistic" in result
    assert "p_value" in result
    assert isinstance(result["t_statistic"], float)
    assert isinstance(result["p_value"], float)


def test_t_test_handles_nan_values():
    group1 = pd.Series([1, 2, np.nan, 4, 5])
    group2 = pd.Series([2, 3, 4, np.nan, 6])

    result = t_test(group1, group2)

    assert not np.isnan(result["t_statistic"])
    assert not np.isnan(result["p_value"])


def test_anova_test_returns_expected_keys():
    df = pd.DataFrame({
        "category": ["A", "A", "B", "B", "C", "C"],
        "value": [1, 2, 3, 4, 5, 6]
    })

    result = anova_test(df, "category", "value")

    assert "f_statistic" in result
    assert "p_value" in result
    assert isinstance(result["f_statistic"], float)
    assert isinstance(result["p_value"], float)


def test_chi_squared_test_returns_expected_keys():
    df = pd.DataFrame({
        "gender": ["M", "F", "M", "F", "M"],
        "choice": ["A", "A", "B", "B", "A"]
    })

    result = chi_squared_test(df, "gender", "choice")

    assert "chi2" in result
    assert "p_value" in result
    assert "dof" in result
    assert "expected" in result

    assert isinstance(result["chi2"], float)
    assert isinstance(result["p_value"], float)
    assert isinstance(result["dof"], int)
    assert isinstance(result["expected"], np.ndarray)


def test_correlation_test_pearson():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })

    result = correlation_test(df, "x", "y", method="pearson")

    assert pytest.approx(result["correlation"], rel=1e-5) == 1.0
    assert result["p_value"] < 0.05


def test_correlation_test_spearman():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50]
    })

    result = correlation_test(df, "x", "y", method="spearman")

    assert pytest.approx(result["correlation"], rel=1e-5) == 1.0
    assert result["p_value"] < 0.05


def test_correlation_test_invalid_method():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6]
    })

    with pytest.raises(ValueError, match="Unsupported method"):
        correlation_test(df, "x", "y", method="kendall")


def test_normality_check_returns_expected_keys():
    df = pd.DataFrame({
        "values": [1.1, 1.9, 3.0, 4.2, 5.1]
    })

    result = normality_check(df, "values")

    assert "shapiro_stat" in result
    assert "p_value" in result

    assert isinstance(result["shapiro_stat"], float)
    assert isinstance(result["p_value"], float)