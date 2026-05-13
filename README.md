# End-to-End Machine Learning Pipeline (MLOps-Inspired System)

An industry-style machine learning project that demonstrates an end-to-end ML workflow including data ingestion, feature engineering, model training, evaluation, and a working inference application.

The project is currently in active development, with MLOps components being progressively integrated into a modular and production-style architecture.

It is designed to showcase practical skills in:
- Unit testing for data science and preprocessing modules
- CI pipeline integration using GitHub Actions
- Machine Learning workflow design
- Data Science and feature engineering
- Software engineering practices for ML systems
- Model deployment (in progress)
- MLOps foundations (MLflow, FastAPI, monitoring modules)

---

## Repository Information

![GitHub stars](https://img.shields.io/github/stars/Shubham-S151/End-to-End-MLOps-Pipeline.svg)
![GitHub forks](https://img.shields.io/github/forks/Shubham-S151/End-to-End-MLOps-Pipeline.svg)
![GitHub issues](https://img.shields.io/github/issues/Shubham-S151/End-to-End-MLOps-Pipeline.svg)
![GitHub license](https://img.shields.io/github/license/Shubham-S151/End-to-End-MLOps-Pipeline.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/Shubham-S151/End-to-End-MLOps-Pipeline.svg)
![CI](https://github.com/Shubham-S151/End-to-End-MLOps-Pipeline/actions/workflows/ci.yml/badge.svg)

---

## Live Application

- Streamlit App: https://end-to-end-mlops-platform.streamlit.app/  
- GitHub Repository: https://github.com/Shubham-S151/End-to-End-MLOps-Pipeline  

---

## Project Status

### Version 1 (Active Development)

The project currently implements a working end-to-end machine learning pipeline with a functional Streamlit-based inference interface.

The MLOps layer is under active development and not fully production-ready yet.

### Completed Components

- Data ingestion and preprocessing pipeline
- Exploratory data analysis (EDA)
- Feature engineering pipeline
- Model training and evaluation pipeline
- Scikit-learn pipeline integration
- Streamlit-based user interface
- Batch prediction using CSV upload
- Single prediction interface
- Modular project structure

### Work in Progress

- FastAPI deployment layer integration
- MLflow experiment tracking stabilization
- ML engineering module (`src/ml_engineering`)
- Monitoring and drift detection integration
- CI/CD pipeline improvements

---

## Problem Statements

### 1. Customer Churn Prediction

Predict whether a telecom customer is likely to leave the service.

Business objectives:
- Improve customer retention
- Reduce churn-related revenue loss
- Enable targeted retention strategies

---

### 2. Credit Card Fraud Detection

Detect fraudulent financial transactions using machine learning models.

Business objectives:
- Reduce financial fraud risk
- Improve transaction security
- Enable scalable fraud detection systems

---

## System Architecture

```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Ingestion
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Scikit-learn Pipeline
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Streamlit Inference Application
```

(MLOps components under development)
- MLflow experiment tracking (implemented in notebooks, integration in pipeline ongoing)
- FastAPI Deployment Layer
- Monitoring and Drift Detection


---

## Project Structure

```bash
end-to-end-mlops-platform/
│
├── api/
│   └── main.py                      # FastAPI service (in progress)
│
├── pipeline/
│   ├── complete_pipeline_v1.py
│   └── training_pipeline_v1.py
│
├── src/
│   ├── common/                      # Shared utilities
│   ├── data_science/                # EDA, preprocessing, feature engineering
│   ├── ml_engineering/              # ML deployment & tracking (in progress)
│   └── Project_Details.md
│
├── streamlit_app/
│   ├── pages/
│   ├── app.py                       # Main UI
│   └── pipeline.pkl                 # Serialized model pipeline
│
├── notebooks/                       # EDA and experimentation
├── tests/                           # Unit tests for modules
├── data/                            # Raw and processed datasets
├── .github/workflows/               # CI pipeline
├── docker/                          # Containerization setup (planned)
├── docs/                            # Internal development notes
├── mlflow.db                        # Experiment tracking database
├── requirements.txt
└── README.md
```

---

## Dataset Information

### Telecom Churn Dataset

* Type: Classification
* Domain: Customer retention analytics

### Credit Card Fraud Dataset

* Type: Imbalanced classification
* Domain: Financial fraud detection

---

## Technology Stack

### Data Science

* Python
* Pandas
* NumPy
* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Machine Learning Pipeline

* Scikit-learn Pipelines
* Feature engineering modules
* Model evaluation framework

### MLOps Components (Partial / In Progress)

* MLflow (experiment tracking)
* FastAPI (model serving layer)
* Evidently AI (monitoring integration - planned)

### Deployment

* Streamlit Cloud (current deployment)

---

## Pipeline Overview

### 1. Data Ingestion

* Load raw datasets
* Train-test split
* Store processed artifacts

### 2. Data Validation

* Schema validation
* Missing value checks
* Data consistency checks

### 3. Feature Engineering

* Encoding categorical variables
* Scaling numerical features
* Feature transformation pipeline

### 4. Model Training

* Multiple ML models trained
* Hyperparameter tuning
* Best model selection

### 5. Model Evaluation

* Performance comparison
* Metric-based model selection

### 6. Streamlit Deployment

* Interactive prediction interface
* Batch prediction support

---

## Skills Demonstrated

### Machine Learning

* End-to-end ML pipeline design
* Feature engineering
* Model training and evaluation
* Handling imbalanced datasets

### Software Engineering

* Modular project architecture
* Reusable code structure
* Separation of concerns

### MLOps (Foundational / In Progress)

* MLflow experiment tracking (in progress)
* FastAPI service design (in progress)
* Monitoring architecture design (planned)

---

## Key Highlights

* Clean modular architecture
* Real-world ML use cases
* End-to-end pipeline implementation
* Working inference application
* CI workflow integration
* Production-style project structure (in development)

---

## Future Improvements

* Full FastAPI deployment integration
* Complete MLflow tracking pipeline
* Docker containerization
* CI/CD automation pipeline
* Cloud deployment (AWS / GCP / Azure)
* Real-time streaming inference system
* Advanced monitoring and drift detection
* Model registry implementation

---

## Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit a pull request

---

## License

This project is licensed under the MIT License.

---

## Contact

Shubham

- LinkedIn: [https://www.linkedin.com/in/shubham-data-science/](https://www.linkedin.com/in/shubham-data-science/)
- GitHub: [https://github.com/Shubham-S151](https://github.com/Shubham-S151)
