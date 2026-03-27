# General Constants
SEED = 42

# Paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/"
LOG_PATH = "logs/"
CONFIG_PATH = "config.yml"

# Environment
ENV = "dev"   # options: dev, staging, prod

# Validation Constants
EXPECTED_SCHEMA = {}
REQUIRED_COLUMNS = []

# Value ranges
AGE_RANGE = (0, 120)
AMOUNT_RANGE = (0, 1e6)

# Allowed categories
GENDER_VALUES = []
CITY_VALUES = []

# Unique keys
UNIQUE_KEYS = []

# Cleaning / ML Constants
# Data Cleaning
MISSING_VALUE_STRATEGY = "median"       # options: mean, median, mode, constant, drop
OUTLIER_METHOD = "IQR"                  # options: IQR, zscore
SCALING_METHOD = "standardization"      # options: normalization, standardization
ENCODING_METHOD = "onehot"              # options: onehot, label

# ML Pipeline
TRAIN_TEST_SPLIT = 0.8
CV_FOLDS = 5

# MLflow / Deployment
MLFLOW_TRACKING_URI = "http://localhost:5000"
DEPLOYMENT_PORT = 8000
MONITORING_THRESHOLD = 0.7
