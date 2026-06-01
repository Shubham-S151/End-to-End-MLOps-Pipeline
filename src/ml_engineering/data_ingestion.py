import os
import numpy as np
import pandas as pd

from src.common.utils import LoggerUtils
from src.common.constants import LOG_PATH

# os.getcwd(os.path.join())
# setup logging
logger = LoggerUtils()
logger.setup_logger("data_ingestion_logs", os.makedirs(os.path.join(LOG_PATH, "data_ingestion_logs.log")))

def import_data(path:str):
    path = path.lower()
    if path.endswith(("csv", "xlsx")):
        print(f"File passed is {path.split(".")[-1]} file")
        logger.log(f"File passed is {path.split(".")[-1]} file")