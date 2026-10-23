import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Załadowanie modelu z rejestru za pomocą aliasu champion
model = mlflow.sklearn.load_model("models:/breast_cancer@champion")

# Pobranie danych i wykonanie predykcji dla pierwszych 10 próbek
X, _ = load_breast_cancer(return_X_y=True)
predictions = model.predict(X[:10])

print(f"Predykcje dla pierwszych dziesieciu próbek: {predictions}")