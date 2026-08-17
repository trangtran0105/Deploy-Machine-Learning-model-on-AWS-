# Telco Customer Churn — End-to-End ML Pipeline on AWS

An end-to-end machine learning project that predicts customer churn for a telecom company, covering the full lifecycle: data validation, feature engineering, hyperparameter-tuned XGBoost training with MLflow tracking, a FastAPI inference service, containerization with Docker, and automated deployment to AWS ECS Fargate via GitHub Actions.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

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
