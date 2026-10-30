import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# Wczytanie zbioru danych
X, y = load_breast_cancer(return_X_y=True)

# Podzial na czesc treningowa (80%) i testowa (20%)
# Argument stratify zapewnia zachowanie proporcji klas w obu czesciach
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Model bazowy z domyslnymi hiperparametrami
baseline = RandomForestClassifier(random_state=42)

# Ocena modelu bazowego metoda pieciokrotnej walidacji krzyzowej
baseline_cv = cross_val_score(baseline, X_train, y_train, cv=5, scoring="accuracy")
print(
    f"Baseline, dokladnosc CV: {baseline_cv.mean():.4f} (odch. std. {baseline_cv.std():.4f})"
)

# Trening na calym zbiorze treningowym i jednokrotna ocena na zbiorze testowym
baseline.fit(X_train, y_train)
baseline_test_acc = baseline.score(X_test, y_test)
print(f"Baseline, dokladnosc na zbiorze testowym: {baseline_test_acc:.4f}")


def objective(trial):
    # Definicja przestrzeni przeszukiwania w paradygmacie define-by-run.
    # Każde wywołanie metody suggest deklaruje jeden hiperparametr i jego zakres.
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }

    # Budowa modelu z zaproponowaną konfiguracją.
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

    # Ocena konfiguracji metodą pięciokrotnej walidacji krzyżowej.
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

    # Zwracana wartość będzie maksymalizowana przez Optunę.
    return score


def objective_przeglad(trial):
    # Liczba całkowita z przedziału domkniętego
    trial.suggest_int("n_estimators", 50, 400)

    # Liczba całkowita z zadanym krokiem
    trial.suggest_int("max_depth", 2, 20, step=2)

    # Liczba zmiennoprzecinkowa z przedziału domkniętego
    trial.suggest_float("subsample", 0.5, 1.0)

    # Liczba zmiennoprzecinkowa próbkowana w skali logarytmicznej
    trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

    # Wybór kategoryczny z listy wartości
    trial.suggest_categorical("criterion", ["gini", "entropy"])


# Utworzenie badania. Dokładność jest maksymalizowana.
study = optuna.create_study(direction="maximize", study_name="rf_tuning")

# Uruchomienie optymalizacji obejmującej 50 prób.
study.optimize(objective, n_trials=50)

print("Najlepsza wartość funkcji celu (dokładność CV):", study.best_value)

print("Najlepsza konfiguracja hiperparametrów:", study.best_params)

# Weryfikacja najlepszego modelu na zbiorze testowym
best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
tuned_test_acc = best_model.score(X_test, y_test)


print(f"Model strojony, dokładność na zbiorze testowym: {tuned_test_acc:.4f}")
print(f"Model bazowy, dokładność na zbiorze testowym: {baseline_test_acc:.4f}")

# Historia optymalizacji
fig1 = plot_optimization_history(study)
fig1.show()

# Ważność hiperparametrów
fig2 = plot_param_importances(study)
fig2.show()
