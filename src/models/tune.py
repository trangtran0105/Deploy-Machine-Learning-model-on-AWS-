import optuna
from xgboost import XGBClassifier
from sklearn.metrics import recall_score

def tune_model(
    X_train, X_test,
    y_train, y_test,
    n_trials: int = 30,
    threshold: float = 0.3,
    scale_pos_weight: float = 1.0
) -> dict:
    """
    Tunes XGBoost using Optuna.
    Returns best_params dict.
    """
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 800),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0, 5),
            "random_state":     42,
            "n_jobs":           -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric":      "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        return recall_score(y_test, y_pred, pos_label=1)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    print(f"✅ Best params: {study.best_params}")
    print(f"✅ Best recall: {study.best_value:.4f}")
    return study.best_params 
