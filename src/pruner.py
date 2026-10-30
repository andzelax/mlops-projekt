import optuna
from optuna.trial import TrialState
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Wczytanie i podział na część treningową oraz testową
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Wydzielenie zbioru walidacyjnego z części treningowej
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)


# 3. Definicja funkcji celu z raportowaniem i przycinaniem
def objective_pruning(trial):
    params = {
        # Celowo duża, stała liczba etapów. O momencie przerwania decyduje pruner.
        "n_estimators": 300,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
    }

    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_tr, y_tr)

    # Ocena modelu etapami, po kolejnych drzewach
    for step, y_pred in enumerate(model.staged_predict(X_val)):
        acc = accuracy_score(y_val, y_pred)
        # Raportowanie wyniku pośredniego
        trial.report(acc, step)

        # Sprawdzenie decyzji prunera i ewentualne przerwanie próby
        if trial.should_prune():
            raise optuna.TrialPruned()

    return acc


# 4. Utworzenie badania z MedianPruner
study_pruned = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,  # pierwsze pięć prób nigdy nie jest przycinanych
        n_warmup_steps=20,  # żadna próba nie jest przycinana przed dwudziestym krokiem
    ),
)

# 5. Uruchomienie 40 prób
study_pruned.optimize(objective_pruning, n_trials=40)

# 6. Zestawienie liczby prób ukończonych i przyciętych
pruned = [t for t in study_pruned.trials if t.state == TrialState.PRUNED]
complete = [t for t in study_pruned.trials if t.state == TrialState.COMPLETE]
print(f"Próby ukończone: {len(complete)}, próby przycięte: {len(pruned)}")
