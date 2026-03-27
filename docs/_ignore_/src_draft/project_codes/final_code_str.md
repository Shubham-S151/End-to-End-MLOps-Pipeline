# 🚀 Production-Level Code Structure: End-to-End MLOps Project

---

## Tree Based Code Structure

---

src/
├── common/
│   ├── utils.py
│   │   ├── setup_logger()
│   │   ├── timeit()
│   │   ├── save_json(), load_json()
│   │   ├── load_config()
│   │   ├── set_seed()
│   │   ├── ensure_dir()
│   │   ├── save_dataframe(), load_dataframe()
│   │   └── timer() context manager
│   ├── constants.py
│   │   ├── SEED, paths, ENV
│   │   ├── EXPECTED_SCHEMA, REQUIRED_COLUMNS, AGE_RANGE, GENDER_VALUES
│   │   ├── MISSING_VALUE_STRATEGY, OUTLIER_METHOD, SCALING_METHOD, ENCODING_METHOD
│   │   ├── PLOT_STYLE, PALETTE, FIGSIZE, HIST_BINS
│   │   └── TRAIN_TEST_SPLIT, CV_FOLDS, MLFLOW_TRACKING_URI, DEPLOYMENT_PORT
│   ├── data_validation.py
│   │   ├── validate_schema()
│   │   ├── check_missing_values()
│   │   ├── check_duplicates()
│   │   ├── check_unique()
│   │   ├── validate_value_ranges()
│   │   ├── validate_categories()
│   │   └── validate_consistency()
│   └── visualization.py
│       ├── UnivariateAnalysis
│       │   ├── density_plot(), box_plot(), distribution_plot(), pie_plot(), plot()
│       ├── BivariateAnalysis
│       │   ├── scatter_plot(), correlation_heatmap(), grouped_boxplot(), crosstab_heatmap()
│       ├── TimeSeriesAnalysis
│       │   ├── line_plot(), rolling_average(), seasonal_decompose(), autocorrelation_plot()
│       └── CustomPlots
│           ├── correlation_heatmap(), pairplot(), target_vs_feature_plot(), geo_plot()
│
├── data_science/
│   ├── data_inspection.py → inspect_head(), inspect_info(), inspect_summary()
│   ├── descriptive_analysis.py → summary_statistics(), correlation_matrix(), grouped_summary()
│   ├── data_cleaning.py
│   │   └── DataCleaning class
│   │       ├── drop_duplicates()
│   │       ├── handle_missing_values()
│   │       ├── normalize_numeric()
│   │       ├── encode_categorical()
│   │       ├── standardize_column_names()
│   │       ├── clean_text_column()
│   │       ├── convert_to_datetime()
│   │       ├── handle_outliers()
│   │       └── get_cleaned_data()
│   ├── inferential_analysis.py → t_test(), chi_square_test(), anova_test()
│   ├── feature_engineering.py → create_interaction_terms(), bin_numeric(), encode_target(), feature_selection()
│   └── preprocessing.py → run_pipeline(), apply_validation(), apply_cleaning(), apply_feature_engineering(), save_preprocessed_data()
│
└── ml_engineering/
    ├── data_ingestion.py → load_data(), split_data()
    ├── model_training.py → train_model(), save_model(), load_model()
    ├── model_evaluation.py → evaluate_model(), plot_confusion_matrix(), plot_roc_curve(), classification_report()
    ├── mlflow_tracking.py → log_params(), log_metrics(), log_model(), set_experiment()
    ├── deployment_fastapi.py → create_app(), predict_endpoint(), health_check()
    └── monitoring_evidently.py → monitor_data_drift(), monitor_model_performance(), generate_monitoring_report()

---

### 📂 src/

Modular codebase for data science + ML engineering pipeline.

---

### 📦 common/

Reusable utilities and shared components.

- **utils.py**

  - `setup_logger(name, log_file=None, level=logging.INFO)`
  - `timeit(func)`
  - `save_json(data, path)`
  - `load_json(path)`
  - `load_config(path="config.yml")`
  - `set_seed(seed=42)`
  - `ensure_dir(path)`
  - `save_dataframe(df, path, format="csv")`
  - `load_dataframe(path, format="csv")`
- **data_validation.py**

  - `validate_schema(df, expected_schema)`
  - `check_missing_values(df)`
  - `check_duplicates(df)`
  - `check_unique(df, cols)`
  - `validate_value_ranges(df, ranges)`
  - `validate_categories(df, col, allowed_values)`
  - `validate_consistency(df, rules)`
- **visualization.py**

  - **UnivariateAnalysis**
    - `density_plot(col)`
    - `box_plot(col)`
    - `distribution_plot(col)`
    - `pie_plot(col)`
    - `plot()`
  - **BivariateAnalysis**
    - `scatter_plot(x, y)`
    - `correlation_heatmap()`
    - `grouped_boxplot(num_col, cat_col)`
    - `crosstab_heatmap(col1, col2)`
  - **TimeSeriesAnalysis**
    - `line_plot(col, time_col)`
    - `rolling_average(col, window)`
    - `seasonal_decompose(col)`
    - `autocorrelation_plot(col)`
  - **CustomPlots**
    - `correlation_heatmap()`
    - `pairplot(cols)`
    - `target_vs_feature_plot(feature)`
    - `geo_plot(loc_cols)`
- **constants.py**

  - General: `SEED`, `RAW_DATA_PATH`, `MODEL_PATH`, `LOG_PATH`
  - Validation: `EXPECTED_SCHEMA`, `REQUIRED_COLUMNS`, `AGE_RANGE`, `GENDER_VALUES`
  - Cleaning: `MISSING_VALUE_STRATEGY`, `OUTLIER_METHOD`, `SCALING_METHOD`, `ENCODING_METHOD`
  - Visualization: `PLOT_STYLE`, `PALETTE`, `FIGSIZE`, `HIST_BINS`
  - ML Engineering: `TRAIN_TEST_SPLIT`, `CV_FOLDS`, `MLFLOW_TRACKING_URI`, `DEPLOYMENT_PORT`, `MONITORING_THRESHOLD`

---

### 📊 data_science/

Modules for EDA, data preparation, and feature engineering.

- **data_inspection.py**

  - `inspect_head(df)`
  - `inspect_info(df)`
  - `inspect_summary(df)`
- **descriptive_analysis.py**

  - `summary_statistics(df)`
  - `correlation_matrix(df)`
  - `grouped_summary(df, col)`
- **data_cleaning.py**

  - **DataCleaning**
    - `drop_duplicates()`
    - `handle_missing_values(strategy="mean")`
    - `normalize_numeric(cols=None)`
    - `encode_categorical(cols=None)`
    - `standardize_column_names()`
    - `clean_text_column(col)`
    - `convert_to_datetime(col, format=None)`
    - `handle_outliers(method="IQR")`
    - `get_cleaned_data()`
- **inferential_analysis.py**

  - `t_test(group1, group2)`
  - `chi_square_test(df, col1, col2)`
  - `anova_test(df, col, group_col)`
- **feature_engineering.py**

  - `create_interaction_terms(df)`
  - `bin_numeric(df, col)`
  - `encode_target(df, target_col)`
  - `feature_selection(df, method)`
- **preprocessing.py**

  - `run_pipeline(df, config)`
  - `apply_validation(df)`
  - `apply_cleaning(df)`
  - `apply_feature_engineering(df)`
  - `save_preprocessed_data(df, path)`

---

### 🤖 ml_engineering/

Modules for ML pipeline, training, deployment, and monitoring.

- **data_ingestion.py**

  - `load_data(path)`
  - `split_data(df, train_size)`
- **model_training.py**

  - `train_model(X_train, y_train, model_type)`
  - `save_model(model, path)`
  - `load_model(path)`
- **model_evaluation.py**

  - `evaluate_model(model, X_test, y_test)`
  - `plot_confusion_matrix(y_true, y_pred)`
  - `plot_roc_curve(y_true, y_pred)`
  - `classification_report(y_true, y_pred)`
- **mlflow_tracking.py**

  - `log_params(params)`
  - `log_metrics(metrics)`
  - `log_model(model)`
  - `set_experiment(name)`
- **deployment_fastapi.py**

  - `create_app()`
  - `predict_endpoint(model)`
  - `health_check()`
- **monitoring_evidently.py**

  - `monitor_data_drift(df)`
  - `monitor_model_performance(y_true, y_pred)`
  - `generate_monitoring_report(df, model)`
