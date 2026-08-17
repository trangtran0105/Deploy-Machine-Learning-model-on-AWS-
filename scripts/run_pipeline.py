#!/usr/bin/env python3
"""
run_pipeline.py
Unified pipeline: load → preprocess → encode target → build features → train (Optuna) → evaluate 

"""

from pyexpat import model
import os, sys, json, time, joblib, argparse
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split

# STAGE 1: DATA LOAD
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data
from src.models.tune import tune_model
from src.models.train import train_model
from src.models.evaluate import evaluate_model 

# ─────────────────────────────────────────────
# STAGE 2: DATA PREPARATION 
# ─────────────────────────────────────────────
def prepare_data(input_path: str, target: str = "Churn"):
    print("🔄 [1/4] Loading data...")
    df = load_data(input_path)
    print(f"   Shape: {df.shape}")

    print("🔍 [2/4] Validating data quality...")
    is_valid, failed = validate_telco_data(df)
    if not is_valid:
        raise ValueError(f"❌ Data quality check failed: {failed}")
    print("✅ Data validation passed.") 

    print("🔧 [3/4] Preprocessing...")
    df = preprocess_data(df, target_col=target)
    print(f"   Shape after preprocess: {df.shape}")

    # Important: Encode target to 0/1
    if df[target].dtype == "object":
        df[target] = df[target].str.strip().map({"No": 0, "Yes": 1})
    df[target] = df[target].astype(int)

    # Sanity check
    assert df[target].isna().sum() == 0, f"{target} has NaN after preprocessing"
    assert set(df[target].unique()) <= {0, 1}, f"{target} is not 0/1"
    print(f"   ✅ Target '{target}' with encode: {df[target].value_counts().to_dict()}")

    print("🛠️  [4/4] Building features...")
    df_enc = build_features(df, target_col=target)

    # Cast bool → int for XGBoost
    for c in df_enc.select_dtypes(include=["bool"]).columns:
        df_enc[c] = df_enc[c].astype(int)
    print(f"   ✅ Features: {df_enc.shape[1]} columns")

    return df_enc, is_valid, failed

# ─────────────────────────────────────────────
# STAGE 3: TRAIN FINAL MODEL AND EVALUATE
# ─────────────────────────────────────────────
def main(args):
    # ── MLflow Setup ──
    mlruns_path = args.mlflow_uri or f"file://{PROJECT_ROOT}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost_optuna")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("n_trials", args.n_trials)

        # ── Prepare data ── 
        df_enc, is_valid, failed = prepare_data(args.input, target=args.target) 
        
        # ── Validate raw data ── 
        print("🔍 Validating data quality...")
        mlflow.log_metric("data_quality_pass", int(is_valid))
        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed: {failed}")
        print("✅ Data validation passed.")

        # ── Save processed data ──
        processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_enc.to_csv(processed_path, index=False)
        print(f"✅ Processed dataset saved → {processed_path}")

        # ── Train/Test Split ──
        print("📊 Splitting data...")
        target = args.target
        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=42      
        )
        print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

        # ── Class imbalance weight ──
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f}")
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))

        # ── Optuna hyperparameter search ──
        print("🔍 Running hyperparameter tuning...")
        best_params = tune_model(
            X_train, X_test,
            y_train, y_test,
            n_trials=args.n_trials,
            threshold=args.threshold,
            scale_pos_weight=scale_pos_weight
        )
        # Log best params
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        print(f"✅ Best params: {best_params}")

        # ── Train final model with best params ──
        print("\n🤖 Training final model with best params...")
        model, train_time = train_model(
            X_train, y_train,
            best_params=best_params,
            scale_pos_weight=scale_pos_weight
        )
        mlflow.log_metric("train_time", train_time)

        # ── Evaluate ──
        print("📊 Evaluating...")
        metrics = evaluate_model(
            model, X_test, y_test,
            threshold=args.threshold
        )
        
        # ── Save artifacts ──
        artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(X.columns)
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        preprocessing_artifact = {"feature_columns": feature_cols, "target": target}
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))

        joblib.dump(model, os.path.join(artifacts_dir, "model.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "model.pkl"))
        print(f"✅ Model saved to artifacts/model.pkl")

        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"\n✅ Model and artifacts are logged in MLflow.")

        print(f"\n⏱️  Conclusion:")
        print(f"   Training time  : {train_time:.2f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified Churn Pipeline")
    p.add_argument("--input",      type=str,   required=True)
    p.add_argument("--target",     type=str,   default="Churn")
    p.add_argument("--threshold",  type=float, default=0.3)
    p.add_argument("--test_size",  type=float, default=0.2)
    p.add_argument("--n_trials",   type=int,   default=30)
    p.add_argument("--experiment", type=str,   default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str,   default=None)
    args = p.parse_args()
    main(args) 
    