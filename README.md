# Telco Customer Churn — End-to-End ML Pipeline

An end-to-end machine learning project that predicts customer churn for a telecom company, covering the full lifecycle: data validation, feature engineering, hyperparameter-tuned XGBoost training with MLflow tracking, and a combined FastAPI + Gradio serving app deployed for free on Render.

🔗 **Live Demo (interactive UI):** https://customer-churn-api-1ggw.onrender.com/ui

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
5. Serves predictions through a combined FastAPI REST endpoint + Gradio web UI, using a recall-optimized decision threshold (0.3) consistent with training evaluation.
6. Runs in a single Docker container, deployed for free on Render with auto-deploy on every push to `main`.

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
FastAPI + Gradio Serving (src/app, src/serving)
   │
   ▼
Docker Image ──▶ Render (auto-deploy on push to main)
```

## Project Structure

```
.
├── artifacts/               # Trained model + preprocessing artifacts
│   ├── model.pkl
│   ├── preprocessing.pkl
│   └── feature_columns.json
├── data/                    # Raw and processed datasets
├── notebooks/                # EDA notebook
├── scripts/                  # Pipeline entrypoints and test scripts
│   ├── run_pipeline.py           # Unified training pipeline (load → preprocess → tune → train → evaluate)
│   ├── prepare_processed_data.py
│   └── test_fastapi.py           # Sample script to hit the deployed /predict endpoint
├── src/
│   ├── app/
│   │   └── app.py             # Combined FastAPI + Gradio entrypoint (serves /predict and /ui)
│   ├── data/                 # Data loading & preprocessing
│   ├── features/             # Feature engineering
│   ├── models/                # Training, tuning (Optuna), evaluation
│   ├── serving/
│   │   └── inference.py       # Single source of truth for model loading + prediction logic
│   └── utils/                 # Data quality validation
├── dockerfile
├── requirements.txt
└── Command.md                 # Useful commands reference
```

## Tech Stack

| Category            | Tools                                  |
|----------------------|-----------------------------------------|
| Language              | Python 3.11                             |
| Modeling             | XGBoost, scikit-learn                   |
| Hyperparameter tuning | Optuna                                  |
| Experiment tracking   | MLflow (training only)                  |
| Data validation       | Great Expectations                       |
| Serving              | FastAPI, Uvicorn                         |
| Demo UI              | Gradio (mounted at `/ui`)                |
| Containerization      | Docker                                   |
| Hosting              | Render (free tier, auto-deploy from GitHub) |
