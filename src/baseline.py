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
