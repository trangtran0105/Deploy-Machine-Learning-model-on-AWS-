import mlflow
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(
    model,
    X_test,
    y_test,
    threshold: float = 0.3
) -> dict:
    """
    Evaluates model and logs metrics to MLflow.
    Returns dict of metrics.
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, proba)

    # Log into MLflow
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1",        f1)
    mlflow.log_metric("roc_auc",   roc_auc)

    print(f"\n🎯 Results:")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1        : {f1:.4f}")
    print(f"   ROC AUC   : {roc_auc:.4f}")
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc
    }