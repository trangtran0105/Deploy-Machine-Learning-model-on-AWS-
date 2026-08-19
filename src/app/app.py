from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys

# Ensure we can import from src/serving when running "uvicorn app.app:app"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict  # our single source of truth for inference

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"status": "ok", "model": "telco-churn"}

# Request schema (same fields you collect in the UI)
class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def api_predict(data: CustomerData):
    try:
        out = predict(data.dict())
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}

# === GRADIO UI with input validation ===
def validate_and_predict(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    # --- Required fields validation ---
    required_fields = {
        "Gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "Phone Service": PhoneService,
        "Multiple Lines": MultipleLines,
        "Internet Service": InternetService,
        "Online Security": OnlineSecurity,
        "Online Backup": OnlineBackup,
        "Device Protection": DeviceProtection,
        "Tech Support": TechSupport,
        "Streaming TV": StreamingTV,
        "Streaming Movies": StreamingMovies,
        "Contract": Contract,
        "Paperless Billing": PaperlessBilling,
        "Payment Method": PaymentMethod,
        "Tenure": tenure,
        "Monthly Charges": MonthlyCharges,
        "Total Charges": TotalCharges,
    }
    for field_name, field_value in required_fields.items():
        if field_value is None or str(field_value).strip() == "":
            raise gr.Error("⚠️ Please fill in all required information")

    # --- Validate tenure ---
    if not str(tenure).strip().isdigit():
        raise gr.Error("⚠️ Tenure must be a whole number")

    # --- Validate monthly / total charges ---
    try:
        monthly_val = float(MonthlyCharges)
    except (TypeError, ValueError):
        raise gr.Error("⚠️ Monthly Charges must be a valid number")

    try:
        total_val = float(TotalCharges)
    except (TypeError, ValueError):
        raise gr.Error("⚠️ Total Charges must be a valid number")

    if total_val < monthly_val:
        raise gr.Error("⚠️ Total charges must be greater than or equal to Monthly charges")

    payload = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": monthly_val,
        "TotalCharges": total_val,
    }
    return predict(payload)

demo = gr.Interface(
    fn=validate_and_predict,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            label="Payment Method"
        ),
        gr.Textbox(label="Tenure (months)", placeholder="e.g. 12"),
        gr.Textbox(label="Monthly Charges ($)", placeholder="e.g. 85"),
        gr.Textbox(label="Total Charges ($)", placeholder="e.g. 1234"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    allow_flagging="never",
    title="🔮 Telco Churn Predictor",
    description="Fill in the customer details to get a churn prediction.",
)

app = gr.mount_gradio_app(app, demo, path="/ui")