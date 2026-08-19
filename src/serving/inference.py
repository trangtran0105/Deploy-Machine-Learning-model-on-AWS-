import os
import json
import joblib
import pandas as pd

# === MODEL LOADING CONFIGURATION ===
# Loads directly from the artifacts/ folder baked into the Docker image
# (see dockerfile: COPY artifacts/ /app/artifacts/)
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")

# CRITICAL: must match the threshold used during training/evaluation
# (see run_pipeline.py / test_pipeline_phase2_modeling.py: THRESHOLD = 0.3)
THRESHOLD = 0.3

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    raise Exception(f"Failed to load model from {MODEL_PATH}: {e}")

# === FEATURE SCHEMA LOADING ===
try:
    with open(FEATURE_COLUMNS_PATH) as f:
        FEATURE_COLS = json.load(f)
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    Must stay in sync with src/features/build_features.py.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip()
                .map(mapping).astype("Int64").fillna(0).astype(int)
            )

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.

    Uses predict_proba with the same THRESHOLD (0.3) applied during training
    evaluation, instead of the model's default 0.5 cutoff, to stay consistent
    with the recall-optimized threshold chosen during tuning.

    Returns:
        "Likely to churn" or "Not likely to churn"
    """
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        proba = model.predict_proba(df_enc)[:, 1]
        prob_value = float(proba[0])
        result = 1 if prob_value >= THRESHOLD else 0
        print(f"Probability: {prob_value:.4f} | Threshold: {THRESHOLD} | Result: {result}")
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"