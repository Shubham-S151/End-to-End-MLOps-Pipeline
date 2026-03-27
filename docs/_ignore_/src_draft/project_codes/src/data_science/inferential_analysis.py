import pandas as pd
from scipy import stats

def t_test(group1: pd.Series, group2: pd.Series) -> dict:
    """
    Perform independent two-sample t-test between two groups.
    Returns t-statistic and p-value.
    """
    t_stat, p_val = stats.ttest_ind(group1.dropna(), group2.dropna())
    return {"t_statistic": t_stat, "p_value": p_val}

def anova_test(df: pd.DataFrame, cat_col: str, num_col: str) -> dict:
    """
    Perform one-way ANOVA test for a numerical column grouped by a categorical column.
    Returns F-statistic and p-value.
    """
    groups = [df[num_col][df[cat_col] == cat].dropna() for cat in df[cat_col].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    return {"f_statistic": f_stat, "p_value": p_val}

def chi_squared_test(df: pd.DataFrame, cat_col1: str, cat_col2: str) -> dict:
    """
    Perform Chi-squared test of independence between two categorical columns.
    Returns chi2 statistic, p-value, degrees of freedom, and expected frequencies.
    """
    contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    return {"chi2": chi2, "p_value": p_val, "dof": dof, "expected": expected}

def correlation_test(df: pd.DataFrame, col1: str, col2: str, method: str = "pearson") -> dict:
    """
    Perform correlation test between two numerical columns.
    Supported methods: 'pearson', 'spearman'.
    Returns correlation coefficient and p-value.
    """
    if method == "pearson":
        corr, p_val = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
    elif method == "spearman":
        corr, p_val = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
    else:
        raise ValueError("Unsupported method. Use 'pearson' or 'spearman'.")
    return {"correlation": corr, "p_value": p_val}

def normality_check(df: pd.DataFrame, col: str) -> dict:
    """
    Check normality of a numerical column using Shapiro-Wilk test.
    Returns statistic and p-value.
    """
    stat, p_val = stats.shapiro(df[col].dropna())
    return {"shapiro_stat": stat, "p_value": p_val}
