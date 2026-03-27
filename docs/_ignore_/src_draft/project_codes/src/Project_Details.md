# Final Project Document: Plug-and-Play End-to-End MLOps Platform

## Project Overview

This project is a **config-driven, plug-and-play End-to-End MLOps Platform** designed to demonstrate industry readiness for Data Scientist and ML Engineer roles. The MVP focuses on **Credit Card Fraud Detection**, but the architecture is extensible to other datasets such as Telecom Churn. The platform automates the standard ML lifecycle steps (data cleaning, transformation, visualization, validation, training, evaluation, deployment, monitoring), while allowing users to plug in domain-specific data processing and feature engineering code.

## Objectives

- Showcase ability to design and implement **production-grade ML pipelines**.
- Demonstrate **MLOps skills**: modular architecture, experiment tracking, deployment, monitoring.
- Provide a **config.yml interface** for easy plug-and-play usage.
- Build a scalable foundation for future cloud-native deployment.

## Features

- **Config-driven pipeline**: Users specify dataset path, schema, model type, hyperparameters.
- **Automated lifecycle steps**:
  - Data cleaning and transformation
  - Basic feature engineering
  - Visualization (EDA plots auto-generated)
  - Statistical tests and validation
  - Report generation (downloadable plots + web view)
- **User customization**:
  - Domain-specific preprocessing
  - Custom feature engineering
  - Upload custom models
- **Model lifecycle management**:
  - Training and evaluation
  - MLflow experiment tracking
  - Best model selection (auto or user override)
- **Deployment**:
  - FastAPI REST API for predictions
- **Monitoring**:
  - Evidently AI for drift detection
- **Automated testing**:
  - Each stage validated before progressing to the next

## Architecture

```plaintext
User Code (Custom Preprocessing + Feature Engineering)
        │
        ▼
Config.yml (Dataset path, schema, model type, hyperparams)
        │
        ▼
Platform Core (Automated)
   ├── Data Cleaning & Transformation
   ├── Basic Feature Engineering
   ├── Visualization (EDA plots)
   ├── Statistical Tests & Validation
   └── Report Generation (downloadable plots + web view)
        │
        ▼
Model Lifecycle
   ├── Model Training (default + user-provided models)
   ├── Model Evaluation (metrics, comparisons)
   ├── MLflow Tracking (experiments, artifacts)
   └── Model Selection (auto or user override)
        │
        ▼
Deployment & Monitoring
   ├── FastAPI REST API (prediction endpoint)
   ├── Custom Model Upload option
   └── Evidently AI (drift detection, monitoring reports)
```

## Repository Structure

```plaintext
├── api/                          # FastAPI app for serving predictions
│   └── main.py
├── data/                         # Raw datasets and data dictionaries
│   └── data descriptions/
├── docker/                       # Dockerfiles and deployment docs
├── docs/                         # Documentation and references
├── notebooks/                    # EDA and analysis notebooks
│   ├── Credit_Card_Fraud_Data/
│   └── Telecom_Churn_Data/
├── pipeline/                     # Orchestration scripts
│   └── training_pipeline.py
├── src/                          # Modular codebase
│   ├── common/                   # Reusable utilities
│   │   ├── utils.py
│   │   ├── data_validation.py
│   │   ├── visualization.py
│   │   └── constants.py
│   ├── data_science/             # Data science modules
│   │   ├── data_inspection.py
│   │   ├── descriptive_analysis.py
│   │   ├── data_cleaning.py
│   │   ├── inferential_analysis.py
│   │   ├── feature_engineering.py
│   │   └── preprocessing.py
│   ├── ml_engineering/           # ML engineering modules
│   │   ├── data_ingestion.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   ├── mlflow_tracking.py
│   │   ├── deployment_fastapi.py
│   │   └── monitoring_evidently.py
├── test/                         # Testing scripts
│   ├── repo_structure.py
│   └── test_env.py
├── requirements.txt              # Dependencies
├── README.md                     # Project overview
└── LICENSE                       # License information
```

## Database Choice

* **MVP:** CSV/Parquet files for ingestion (fraud dataset ~555k rows, 22 columns).
* **Future:** SQL (PostgreSQL/MySQL) or cloud data lake (AWS S3, GCP Storage) for scalability.

## Deployment Platforms

* **Render:** Best for FastAPI + Docker, free tier available.
* **Railway.app:** Simple GitHub integration, free tier.
* **Hugging Face Spaces:** Great for ML demos, widely recognized in ML community.
* **Deta Space:** Lightweight, free hosting for Python APIs.

## Status

* **Development completed:** MVP pipeline, modular codebase, config-driven design.
* **Deployment pending:** CI/CD workflows and cloud deployment scripts included to demonstrate skills, but live deployment not yet active.

## Future Improvements

* CI/CD integration with GitHub Actions
* Cloud-native deployment (AWS/GCP/Azure)
* Feature store integration (Feast)
* Streaming ingestion (Kafka/Kinesis)
* Advanced monitoring (Prometheus/Grafana)
