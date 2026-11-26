import os
from pathlib import Path

import optuna
import yaml
from optuna_integration.mlflow import MLflowCallback

from src.data import load_data, split_data
from src.evaluate import evaluate_model
from src.train import train_model

SCIEZKA_KONFIGURACJI = Path(__file__).resolve().parent.parent / "config.yaml"
with open(SCIEZKA_KONFIGURACJI, encoding="utf-8") as plik:
    KONFIGURACJA = yaml.safe_load(plik)

TRACKING_URI = os.getenv("MLFLOW_URI", KONFIGURACJA["mlflow"]["tracking_uri"])
STORAGE = KONFIGURACJA["tuning"]["storage"]
STUDY_NAME = KONFIGURACJA["tuning"]["study_name"]
N_TRIALS = KONFIGURACJA["tuning"]["n_trials"]
N_ESTIMATORS_MIN, N_ESTIMATORS_MAX = KONFIGURACJA["tuning"]["n_estimators_range"]
MAX_DEPTH_MIN, MAX_DEPTH_MAX = KONFIGURACJA["tuning"]["max_depth_range"]


def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", N_ESTIMATORS_MIN, N_ESTIMATORS_MAX
        ),
        "max_depth": trial.suggest_int("max_depth", MAX_DEPTH_MIN, MAX_DEPTH_MAX),
    }
    model = train_model(X_train, y_train, params)
    return evaluate_model(model, X_test, y_test)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    mlflow_callback = MLflowCallback(
        tracking_uri=TRACKING_URI,
        metric_name="accuracy",
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test),
        n_trials=N_TRIALS,
        callbacks=[mlflow_callback],
    )

    print(f"Najlepsze parametry: {study.best_params}")
    print(f"Najlepsza trafnosc: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
