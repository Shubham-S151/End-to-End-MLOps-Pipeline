# 📂 Project Structure & Module Guide

---

## Code Tree

src/
├── common/
│   ├── utils.py
│   │   ├── setup_logger()
│   │   ├── timeit()
│   │   ├── save_json(), load_json()
│   │   └── load_config(), set_seed(), ensure_dir()
│   ├── constants.py
│   │   ├── General constants (SEED, paths)
│   │   ├── Validation constants (schema, ranges)
│   │   └── Cleaning/ML constants (strategies, split ratios)
│   └── data_validation.py
│       ├── validate_schema()
│       ├── check_missing_values()
│       ├── check_duplicates()
│       └── validate_categories(), validate_value_ranges()
│
├── data_science/
│   ├── data_inspection.py → start here (basic df inspection)
│   ├── descriptive_analysis.py → summary stats, correlations
│   ├── data_cleaning.py → cleaning class (duplicates, missing, outliers)
│   ├── inferential_analysis.py → statistical tests
│   ├── feature_engineering.py → feature creation/selection
│   └── preprocessing.py → orchestrates validation → cleaning → features
│
└── ml_engineering/
    ├── data_ingestion.py → load/split data
    ├── model_training.py → train/save/load models
    ├── model_evaluation.py → metrics + plots
    ├── mlflow_tracking.py → log params/metrics/models
    ├── deployment_fastapi.py → API endpoints
    └── monitoring_evidently.py → drift/performance monitoring

---

## src/

Modular codebase for end-to-end MLOps pipeline.

---

### common/

Reusable utilities and shared components.

- **utils.py**

  - `setup_logger(name, log_file=None, level=logging.INFO)`
  - `timeit(func)`
  - `save_json(data, path)`
  - `load_json(path)`
  - (Optional) `load_config(path="config.yml")`
  - (Optional) `set_seed(seed=42)`
  - (Optional) `ensure_dir(path)`
  - (Optional) `save_dataframe(df, path, format="csv")`
  - (Optional) `load_dataframe(path, format="csv")`
- **data_validation.py**

  - `validate_schema(df, expected_schema)`
  - `check_missing_values(df)`
  - `check_duplicates(df)`
  - (Optional) `check_unique(df, cols)`
  - (Optional) `validate_value_ranges(df, ranges)`
  - (Optional) `validate_categories(df, col, allowed_values)`
- **visualization.py**

  - **UnivariateAnalysis**
    - `density_plot(col)`
    - `box_plot(col)`
    - `distribution_plot(col)`
    - `pie_plot(col)`
    - `plot()`
  - **BivariateAnalysis**
    - (planned) `scatter_plot(x, y)`
    - (planned) `correlation_heatmap()`
    - (planned) `grouped_boxplot(num_col, cat_col)`
  - **TimeSeriesAnalysis**
    - (planned) `line_plot(col, time_col)`
    - (planned) `rolling_average(col, window)`
    - (planned) `seasonal_decompose(col)`
  - **CustomPlots**
    - (planned) `correlation_heatmap()`
    - (planned) `pairplot(cols)`
    - (planned) `target_vs_feature_plot(feature)`
- **constants.py**

  - General: `SEED`, `RAW_DATA_PATH`, `MODEL_PATH`, `LOG_PATH`
  - Validation: `EXPECTED_SCHEMA`, `REQUIRED_COLUMNS`, `AGE_RANGE`, `GENDER_VALUES`
  - Cleaning: `MISSING_VALUE_STRATEGY`, `OUTLIER_METHOD`, `SCALING_METHOD`, `ENCODING_METHOD`
  - Visualization: `PLOT_STYLE`, `PALETTE`, `FIGSIZE`, `HIST_BINS`
  - ML Engineering: `TRAIN_TEST_SPLIT`, `CV_FOLDS`, `MLFLOW_TRACKING_URI`, `DEPLOYMENT_PORT`

---

### data_science/

Modules for EDA, data preparation, and feature work.

- **data_inspection.py**

  - (planned) `inspect_head(df)`
  - (planned) `inspect_info(df)`
  - (planned) `inspect_summary(df)`
- **descriptive_analysis.py**

  - (planned) `summary_statistics(df)`
  - (planned) `correlation_matrix(df)`
  - (planned) `grouped_summary(df, col)`
- **data_cleaning.py**

  - **DataCleaning**
    - `drop_duplicates()`
    - `handle_missing_values(strategy="mean")`
    - `normalize_numeric(cols=None)`
    - `encode_categorical(cols=None)`
    - `standardize_column_names()`
    - `clean_text_column(col)`
    - `convert_to_datetime(col, format=None)`
    - `get_cleaned_data()`
- **inferential_analysis.py**

  - (planned) `t_test(group1, group2)`
  - (planned) `chi_square_test(df, col1, col2)`
  - (planned) `anova_test(df, col, group_col)`
- **feature_engineering.py**

  - (planned) `create_interaction_terms(df)`
  - (planned) `bin_numeric(df, col)`
  - (planned) `encode_target(df, target_col)`
- **preprocessing.py**

  - (planned) `run_pipeline(df, config)`
  - (planned) `apply_validation(df)`
  - (planned) `apply_cleaning(df)`
  - (planned) `apply_feature_engineering(df)`

---

### ml_engineering/

Modules for ML pipeline, training, deployment, and monitoring.

- **data_ingestion.py**

  - (planned) `load_data(path)`
  - (planned) `split_data(df, train_size)`
- **model_training.py**

  - (planned) `train_model(X_train, y_train, model_type)`
  - (planned) `save_model(model, path)`
- **model_evaluation.py**

  - (planned) `evaluate_model(model, X_test, y_test)`
  - (planned) `plot_confusion_matrix(y_true, y_pred)`
  - (planned) `plot_roc_curve(y_true, y_pred)`
- **mlflow_tracking.py**

  - (planned) `log_params(params)`
  - (planned) `log_metrics(metrics)`
  - (planned) `log_model(model)`
- **deployment_fastapi.py**

  - (planned) `create_app()`
  - (planned) `predict_endpoint(model)`
- **monitoring_evidently.py**

  - (planned) `monitor_data_drift(df)`
  - (planned) `monitor_model_performance(y_true, y_pred)`
