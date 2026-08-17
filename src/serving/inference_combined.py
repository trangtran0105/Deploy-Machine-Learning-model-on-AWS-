import os
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import joblib
from xgboost import XGBClassifier 

# === CONFIG ===
THRESHOLD = 0.3

# === LOAD MODEL ===
try:
    model = joblib.load("artifacts/model.pkl")
    print("✅ Model loaded from artifacts/model.pkl")
except Exception as e:
    raise Exception(f"Failed to load model: {e}")

# === LOAD FEATURES ===
try:
    # ✅ Đọc từ artifacts/
    with open("artifacts/feature_columns.json") as f:
        FEATURE_COLS = json.load(f)
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === BINARY MAPPINGS ===
BINARY_MAP = {
    "gender":          {"Female": 0, "Male": 1},
    "Partner":         {"No": 0, "Yes": 1},
    "Dependents":      {"No": 0, "Yes": 1},
    "PhoneService":    {"No": 0, "Yes": 1},
    "PaperlessBilling":{"No": 0, "Yes": 1},
}
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# === TRANSFORM ===
def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().map(mapping)\
                    .astype("Int64").fillna(0).astype(int)

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df

# === PREDICT ===
def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    # ✅ Dùng predict_proba với threshold 0.3
    try:
        proba = model.predict_proba(df_enc)[:, 1]
        prob_value = float(proba[0])
        result = 1 if prob_value >= THRESHOLD else 0
        print(f"Probability: {prob_value:.4f} | Result: {result}")

    except Exception as e:
        raise Exception(f"Prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"

# === FASTAPI ===
app = FastAPI(title="Telco Churn Prediction API")

class CustomerData(BaseModel):
    gender: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PhoneService: str
    Partner: str
    Dependents: str
    PaperlessBilling: str
    MultipleLines: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    PaymentMethod: str

@app.get("/")
def health_check():
    return {"status": "ok", "model": "telco-churn"}

@app.post("/predict")
def predict_endpoint(customer: CustomerData):
    result = predict(customer.dict())
    return {"prediction": result}

# === GRADIO ===
# ✅ 
def validate_and_predict(
    gender, tenure, monthly_charges, total_charges,
    contract, internet_service, phone_service,
    partner, dependents, paperless_billing,
    multiple_lines, online_security, online_backup,
    device_protection, tech_support,
    streaming_tv, streaming_movies, payment_method
):
    # Required fields validation 
    required_fields = {
        "Gender": gender,
        "Tenure": tenure,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "Contract": contract,
        "Internet Service": internet_service,
        "Phone Service": phone_service,
        "Partner": partner,
        "Dependents": dependents,
        "Paperless Billing": paperless_billing,
        "Multiple Lines": multiple_lines,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Payment Method": payment_method,
    }

    for field_name, field_value in required_fields.items():
        if field_value is None or str(field_value).strip() == "":
            raise gr.Error("⚠️ Please fill in all required information") 
        
    # Validate tenure
    if not str(tenure).strip().isdigit():
        raise gr.Error("⚠️ Value must be a number with a maximum of 2 characters")
    if len(str(tenure).strip()) > 2:
        raise gr.Error("⚠️ Value must be a number with a maximum of 2 characters")

    # Validate monthly_charges
    try:
        monthly_val = float(monthly_charges)
    except:
        raise gr.Error("⚠️ Value must be a number with a maximum of 3 characters")
    if len(str(monthly_charges).strip().replace(".", "")) > 3:
        raise gr.Error("⚠️ Value must be a number with a maximum of 3 characters")

    # Validate total_charges
    try:
        total_val = float(total_charges)
    except:
        raise gr.Error("⚠️ Value must be a number with a maximum of 4 characters")
    if len(str(total_charges).strip().replace(".", "")) > 4:
        raise gr.Error("⚠️ Value must be a number with a maximum of 4 characters")

    # Validate total >= monthly
    if total_val < monthly_val:
        raise gr.Error("⚠️ Total charges must be greater than or equal to Monthly charges")

    # Predict 
    input_dict = {
        "gender": gender,
        "tenure": int(tenure),
        "MonthlyCharges": monthly_val,
        "TotalCharges": total_val,
        "Contract": contract,
        "InternetService": internet_service,
        "PhoneService": phone_service,
        "Partner": partner,
        "Dependents": dependents,
        "PaperlessBilling": paperless_billing,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PaymentMethod": payment_method,
    }
    return predict(input_dict)


# ✅ Thêm demo mới này
demo = gr.Interface(
    fn=validate_and_predict,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Textbox(label="Tenure (months)", placeholder="e.g. 12"),
        gr.Textbox(label="Monthly Charges ($)", placeholder="e.g. 85"),
        gr.Textbox(label="Total Charges ($)", placeholder="e.g. 1234"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    allow_flagging="never",
    title="🔮 Telco Customer Churn Prediction",
    description="Fill in the customer details and click 'Submit' to see if they are likely to churn."
)

# Mount Gradio vào FastAPI
app = gr.mount_gradio_app(app, demo, path="/demo")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)