import os
import numpy as np
import pandas as pd

from src.common.utils import LoggerUtils
from src.common.constants import LOG_PATH

# Setup logging
logger = LoggerUtils()

log_file = os.path.join(LOG_PATH, "data_ingestion_logs.log")
os.makedirs(LOG_PATH, exist_ok=True)

logger.setup_logger("data_ingestion_logs", log_file)


def import_data(path: str) -> pd.DataFrame:
    """
    Import tabular data from CSV or Excel files.

    Args:
        path (str): Path to the file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        extension = path.lower().split(".")[-1]

        if extension == "csv":
            logger.log("CSV file detected.")
            print("CSV file detected.")

            df = pd.read_csv(path)

        elif extension in ["xlsx", "xls"]:
            logger.log("Excel file detected.")
            print("Excel file detected.")

            df = pd.read_excel(path)

        else:
            raise ValueError(
                f"Unsupported file format '{extension}'. "
                "Only CSV and Excel files are supported."
            )

        logger.log(f"Successfully loaded data from {path}")
        logger.log(f"Data shape: {df.shape}")

        print(f"Data shape: {df.shape}")

        return df

    except Exception as e:
        logger.log(f"Error loading data: {str(e)}")
        raise


# Example usage
if __name__ == "__main__":
    df = import_data("data/sample.csv")
    print(df.head())