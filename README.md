# Working on it ......
---

# Current Progress (v1)

**App link:**  [Click Here](https://end-to-end-mlops-platform.streamlit.app/)\
**Project link:** [Click Here](https://github.com/Shubham-S151/End-to-End-MLOps-Pipeline)

The project has successfully reached its first deployed version with core functionality implemented and validated.

## Completed
- End-to-end ML pipeline development (data preprocessing → model → inference)
- Fraud detection model training and evaluation
- Streamlit web application for user interaction
- Batch prediction using CSV upload
- Single transaction prediction interface
- Model integration with deployed UI
- Initial deployment on Streamlit Cloud

---
# End-to-End MLOps Platform

An industry-level **End-to-End Machine Learning Platform** that demonstrates the complete lifecycle of ML systems — from data ingestion and analysis to model deployment and monitoring.

This project is designed to showcase **Data Science + Machine Learning Engineering + MLOps** skills in a production-style architecture.

---

## Problem Statements

This platform supports multiple real-world ML problems:

1. **Customer Churn Prediction**
   - Predict whether a customer will leave a telecom service.

2. **Credit Card Fraud Detection**
   - Detect fraudulent financial transactions.

---

## Project Architecture

```md
Raw Data
↓
EDA (Notebooks)
↓
Data Ingestion
↓
Data Validation
↓
Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
MLflow Tracking
↓
Pipeline Automation
↓
FastAPI Deployment
↓
Monitoring (Evidently)
```

---

## Project Structure

```md
end-to-end-mlops-platform/
│
├── src/
│ ├── data_ingestion/
│ ├── data_validation/
│ ├── feature_engineering/
│ ├── model_training/
│ └── model_evaluation/
│
├── pipeline/
│ └── training_pipeline.py
│
├── notebooks/
│ ├── EDA_telecom_churn.ipynb
│ └── EDA_fraud_transactions.ipynb
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── external/
│
├── api/
│ └── main.py
│
├── docker/
│
├── artifacts/
│
├── requirements.txt
├── README.md
└── LICENSE
```


---

## Datasets Used

### 1. Telecom Customer Churn Dataset
- Business problem: Customer retention
- Type: Classification

### 2. Credit Card Fraud Detection Dataset
- Business problem: Fraud detection
- Type: Imbalanced classification

---

## Tech Stack

### Data Science
- Python
- Pandas
- NumPy
- Scikit-learn

### Visualization
- Matplotlib
- Seaborn

### MLOps & Engineering
- MLflow (Experiment Tracking)
- FastAPI (Model Serving)
- Apache Airflow (Pipeline Orchestration)
- Evidently AI (Monitoring)

### Deployment
- Docker

---

## Pipeline Stages

### 1. Data Ingestion
- Load raw datasets
- Train-test split
- Store artifacts

### 2. Data Validation
- Schema validation
- Missing value checks
- Data consistency checks

### 3. Feature Engineering
- Encoding categorical variables
- Scaling numerical features
- Creating derived features

### 4. Model Training
- Train multiple ML models
- Hyperparameter tuning

### 5. Model Evaluation
- Compare models using metrics
- Select best-performing model

### 6. Experiment Tracking
- Log experiments using MLflow
- Track metrics and parameters

### 7. Deployment
- Serve model using FastAPI
- REST API endpoint for predictions

### 8. Monitoring
- Detect data drift
- Monitor model performance

---

## How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/Shubham-S151/end-to-end-mlops-platform.git
cd end-to-end-mlops-platform
```
### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run Training Pipeline
```bash
python pipeline/training_pipeline.py
```
### 5. Run API
```bash
uvicorn api.main:app --reload
```
## Skills Demonstrated
### Data Scientist Skills

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building & Evaluation
- Handling Imbalanced Data

### ML Engineer Skills

- Pipeline Design
- Modular Code Architecture
- Experiment Tracking
- Model Deployment

### MLOps Skills

- End-to-End ML Workflow
- API Development
- Containerization
- Monitoring & Drift Detection

### Key Highlights

- Modular and scalable ML pipeline
- Supports multiple datasets and use cases
- Production-style project structure
- End-to-end automation from data to deployment

### Future Improvements

- CI/CD integration
- Cloud deployment (AWS/GCP/Azure)
- Real-time streaming pipeline
- Advanced model monitoring

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Feel free to fork the repo and submit pull requests.

## Contact

For queries or collaboration, connect via:

- [Linkedin](https://www.linkedin.com/in/shubham-data-science/)
- [GitHub](https://github.com/Shubham-S151)


---
