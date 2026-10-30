import mlflow
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "lab3-optuna-nested"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


def objective(trial: optuna.Trial) -> float:
    with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": 42,
        }

        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_accuracy = float(scores.mean())

        mlflow.log_metric("cv_accuracy", cv_accuracy)

        return cv_accuracy


with mlflow.start_run(run_name="optuna-study-parent") as parent_run:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    mlflow.log_metric("best_cv_accuracy", study.best_value)
    best_params_prefixed = {f"best_{k}": v for k, v in study.best_params.items()}
    mlflow.log_params(best_params_prefixed)

    print("=" * 60)
    print("ZAKOŃCZONO BADANIE Z URUCHOMIENIAMI ZAGNIEŻDŻONYMI")
    print("=" * 60)
    print(f"Parent Run ID: {parent_run.info.run_id}")
    print(f"Najlepsza wartość (cv_accuracy): {study.best_value:.4f}")
    print(f"Najlepsze parametry: {study.best_params}")
