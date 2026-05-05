import pytest
import pandas as pd
import matplotlib

# Use non-interactive backend for tests
matplotlib.use("Agg")

from your_module_name import (
    summary_statistics,
    plot_histograms,
    plot_boxplots,
    plot_countplots,
    correlation_heatmap,
    pairwise_scatter,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [5, 4, 3, 2, 1],
        "cat1": ["A", "B", "A", "B", "C"]
    })


def test_summary_statistics(sample_df):
    result = summary_statistics(sample_df)

    assert isinstance(result, pd.DataFrame)
    assert "num1" in result.columns
    assert "mean" in result.index

    # Check a known value
    assert result.loc["mean", "num1"] == pytest.approx(3.0)


def test_plot_histograms(sample_df):
    figs = plot_histograms(sample_df, ["num1", "num2"], bins=10)

    assert isinstance(figs, dict)
    assert set(figs.keys()) == {"num1", "num2"}

    for fig in figs.values():
        assert fig is not None
        assert hasattr(fig, "axes")


def test_plot_boxplots(sample_df):
    figs = plot_boxplots(sample_df, ["num1", "num2"])

    assert isinstance(figs, dict)
    assert "num1" in figs
    assert "num2" in figs

    for fig in figs.values():
        assert fig is not None
        assert hasattr(fig, "axes")


def test_plot_countplots(sample_df):
    figs = plot_countplots(sample_df, ["cat1"])

    assert isinstance(figs, dict)
    assert "cat1" in figs

    fig = figs["cat1"]
    assert fig is not None
    assert hasattr(fig, "axes")


def test_correlation_heatmap(sample_df):
    fig = correlation_heatmap(sample_df[["num1", "num2"]])

    assert fig is not None
    assert hasattr(fig, "axes")
    assert len(fig.axes) > 0


def test_pairwise_scatter(sample_df):
    fig = pairwise_scatter(sample_df, ["num1", "num2"])

    # seaborn pairplot returns a PairGrid
    from seaborn.axisgrid import PairGrid
    assert isinstance(fig, PairGrid)