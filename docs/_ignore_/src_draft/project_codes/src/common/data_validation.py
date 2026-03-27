import pandas as pd

# Schema Validation
def validate_schema(df: pd.DataFrame, expected_schema: dict) -> bool:
    """
    Validate dataframe schema against expected column names and dtypes.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        expected_schema (dict): Dictionary of {column_name: expected_dtype}.
    
    Returns:
        bool: True if schema is valid, raises error otherwise.
    """
    for col, dtype in expected_schema.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        if str(df[col].dtype) != str(dtype):
            raise TypeError(f"Column {col} has dtype {df[col].dtype}, expected {dtype}")
    return True

# Missing Values
def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Return percentage of missing values per column.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        dict: {column_name: missing_percentage}.
    """
    return df.isnull().mean().to_dict()

# Duplicate Rows
def check_duplicates(df: pd.DataFrame) -> int:
    """
    Return number of duplicate rows.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        int: Count of duplicate rows.
    """
    return df.duplicated().sum()

# Unique Values
def check_unique(df: pd.DataFrame, cols: list) -> dict:
    """
    Check if specified columns have unique values.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        cols (list): List of column names to check.
    
    Returns:
        dict: {column_name: True/False}.
    """
    return {col: df[col].is_unique for col in cols}

# Value Range Validation
def validate_value_ranges(df: pd.DataFrame, ranges: dict) -> dict:
    """
    Validate numeric columns fall within expected ranges.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        ranges (dict): {column_name: (min_val, max_val)}.
    
    Returns:
        dict: {column_name: count_of_invalid_rows}.
    """
    results = {}
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            invalid = df[(df[col] < min_val) | (df[col] > max_val)]
            results[col] = len(invalid)
    return results

# Category Validation
def validate_categories(df: pd.DataFrame, col: str, allowed_values: list) -> bool:
    """
    Check if categorical column contains only allowed values.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        col (str): Column name to validate.
        allowed_values (list): List of allowed category values.
    
    Returns:
        bool: True if valid, raises ValueError otherwise.
    """
    invalid = set(df[col].dropna().unique()) - set(allowed_values)
    if invalid:
        raise ValueError(f"Invalid values in {col}: {invalid}")
    return True
