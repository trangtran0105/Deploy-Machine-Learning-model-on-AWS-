# Telco Customer Churn — End-to-End ML Pipeline on AWS

An end-to-end machine learning project that predicts customer churn for a telecom company, covering the full lifecycle: data validation, feature engineering, hyperparameter-tuned XGBoost training with MLflow tracking, a FastAPI inference service, containerization with Docker, and automated deployment to AWS ECS Fargate via GitHub Actions.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Running the Training Pipeline](#running-the-training-pipeline)
- [Serving the Model](#serving-the-model)
- [Docker](#docker)
- [CI/CD & Deployment](#cicd--deployment)
- [License](#license)

## Overview

Customer churn is one of the most costly problems for subscription-based businesses. This project builds a reproducible pipeline that:

1. Loads and validates raw telco customer data (data quality checks with Great Expectations).
2. Cleans and preprocesses the data, encoding the target and engineering features (one-hot encoding, binary encoding).
3. Trains an XGBoost classifier, tuning hyperparameters with Optuna to maximize recall on the churn class under class imbalance.
4. Tracks every experiment (parameters, metrics, artifacts) with MLflow.
5. Serves predictions through a FastAPI REST endpoint, containerized with Docker.
6. Deploys automatically to AWS ECS Fargate on every push to `main`, via GitHub Actions.
7. Includes a lightweight Gradio demo app for interactive testing.

## Architecture

```
Raw CSV
   │
   ▼
Data Loading (src/data/load_data.py)
   │
   ▼
Data Validation (src/utils/validate_data.py)  — Great Expectations
   │
   ▼
Preprocessing (src/data/preprocess.py)
   │
   ▼
Feature Engineering (src/features/build_features.py)
   │
   ▼
Train / Test Split
   │
   ▼
Hyperparameter Tuning (src/models/tune.py) — Optuna
   │
   ▼
Model Training (src/models/train.py) — XGBoost
   │
   ▼
Evaluation (src/models/evaluate.py) — precision, recall, F1, ROC-AUC
   │
   ▼
Artifacts saved (model.pkl, preprocessing.pkl, feature_columns.json)
   │        + logged to MLflow
   ▼
FastAPI Serving (src/serving) ──▶ Docker Image ──▶ Amazon ECR
                                                        │
                                                        ▼
                                              AWS ECS Fargate (auto-deployed via GitHub Actions)
```

## Project Structure

```
.
├── .github/workflows/       # CI/CD pipeline (deploy.yml)
├── .gradio/flagged/         # Gradio demo flagged data
├── artifacts/               # Trained model + preprocessing artifacts
│   ├── model.pkl
│   ├── preprocessing.pkl
│   └── feature_columns.json
├── data/                    # Raw and processed datasets
├── flagged/                 # Additional flagged/demo data
├── notebooks/                # EDA notebook
├── scripts/                  # Pipeline entrypoints and test scripts
│   ├── run_pipeline.py           # Unified training pipeline (load → preprocess → tune → train → evaluate)
│   ├── prepare_processed_data.py
│   └── test_fastapi.py           # Sample script to hit the /predict endpoint
├── src/
│   ├── app/                  # Gradio demo application
│   ├── data/                 # Data loading & preprocessing
│   ├── features/             # Feature engineering
│   ├── models/                # Training, tuning (Optuna), evaluation
│   ├── serving/               # FastAPI inference service
│   └── utils/                 # Data quality validation
├── dockerfile
├── requirements.txt
├── task-definition.json      # ECS Fargate task definition
└── Command.md                 # Useful commands reference
```

## Tech Stack

| Category            | Tools                                  |
|----------------------|-----------------------------------------|
| Language              | Python 3.11                             |
| Modeling             | XGBoost, scikit-learn                   |
| Hyperparameter tuning | Optuna                                  |
| Experiment tracking   | MLflow                                  |
| Data validation       | Great Expectations                       |
| Serving              | FastAPI, Uvicorn                         |
| Demo UI              | Gradio                                   |
| Containerization      | Docker                                   |
| CI/CD                | GitHub Actions                            |
| Cloud infrastructure  | AWS ECS Fargate, Amazon ECR              |

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- Docker (optional, for containerized runs)
- AWS CLI configured (only needed for deployment)

### Installation

```bash
git clone https://github.com/trangtran0105/Deploy-Machine-Learning-model-on-AWS-.git
cd Deploy-Machine-Learning-model-on-AWS-
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Running the Training Pipeline

Run the full pipeline (data loading → validation → preprocessing → feature engineering → Optuna tuning → training → evaluation → MLflow logging):

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --n_trials 30
```

Key arguments:

| Argument       | Default        | Description                                  |
|----------------|----------------|-----------------------------------------------|
| `--input`      | *(required)*   | Path to the raw CSV file                       |
| `--target`     | `Churn`        | Target column name                             |
| `--threshold`  | `0.3`          | Classification threshold                        |
| `--test_size`  | `0.2`          | Train/test split ratio                          |
| `--n_trials`   | `30`           | Number of Optuna trials                          |
| `--experiment` | `Telco Churn`  | MLflow experiment name                          |

View experiment results with MLflow's UI:

```bash
mlflow ui
```

## Serving the Model

Start the FastAPI inference server locally:

```bash
python -m uvicorn serving.inference_combined:app --host 0.0.0.0 --port 8000
```

Test the `/predict` endpoint with the provided sample script:

```bash
python scripts/test_fastapi.py
```

Example request body:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 350.75
}
```

## Docker

Build and run the serving container locally:

```bash
docker build -t telco-churn-ml .
docker run -p 8000:8000 telco-churn-ml
```

The API will be available at `http://localhost:8000`.

## CI/CD & Deployment

Every push to `main` triggers `.github/workflows/deploy.yml`, which:

1. Builds the Docker image and tags it with the commit SHA.
2. Pushes the image to Amazon ECR.
3. Updates the ECS task definition (`task-definition.json`) with the new image.
4. Deploys the updated task definition to the `telco-churn-service` on the `telco-churn-cluster` (ECS Fargate, `ap-southeast-1`).

Required GitHub Secrets:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## License

This project is provided for educational and portfolio purposes. Add a license of your choice (e.g., MIT) if you plan to open-source it further.
