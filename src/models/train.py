import time
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier

def train_model(
    X_train, y_train,
    best_params: dict,
    scale_pos_weight: float = 1.0
):
    """
    Trains XGBoost with best_params.
    Returns trained model and train_time.
    """
    # Merge best_params với tham số cố định
    final_params = {
        **best_params,
        "random_state":     42,
        "n_jobs":           -1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric":      "logloss",
    }

    model = XGBClassifier(**final_params)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"✅ Model trained in {train_time:.2f}s")
    return model, train_time