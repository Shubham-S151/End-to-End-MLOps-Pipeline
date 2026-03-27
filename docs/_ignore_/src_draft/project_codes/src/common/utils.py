from contextlib import contextmanager
import pandas as pd
import numpy as np
import logging
import random
import yaml
import time
import json
import os

class LoggerUtils:
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level=logging.INFO):
        """Set up a logger with console and optional file output."""
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger

class TimingUtils:
    @staticmethod
    def timeit(func):
        """Decorator to measure execution time of functions."""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} executed in {end - start:.2f}s")
            return result
        return wrapper

    @staticmethod
    @contextmanager
    def timer(name="block"):
        """Context manager to measure execution time of code blocks."""
        start = time.time()
        yield
        end = time.time()
        print(f"{name} executed in {end - start:.2f}s")

class ConfigUtils:
    @staticmethod
    def load_config(path="config.yml"):
        """Load YAML config file."""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def set_seed(seed=42):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

class FileUtils:
    @staticmethod
    def ensure_dir(path):
        """Create directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def save_json(data, path):
        """Save dictionary as JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_json(path):
        """Load JSON file into dictionary."""
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def safe_load_json(path, default=None):
        """Load JSON safely, return default if file missing or corrupted."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    @staticmethod
    def save_dataframe(df, path, format="csv"):
        """Save DataFrame to CSV or Parquet."""
        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "parquet":
            df.to_parquet(path, index=False)

    @staticmethod
    def load_dataframe(path, format="csv"):
        """Load DataFrame from CSV or Parquet."""
        if format == "csv":
            return pd.read_csv(path)
        elif format == "parquet":
            return pd.read_parquet(path)
