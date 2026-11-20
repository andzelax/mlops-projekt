from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import LICZBA_CECH, app


@pytest.fixture
def client():
    """Podstawienie obiektu zastepczego do app.state.model uniezaleznia testy od MLflow."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=int)
    mock_model.predict_proba.side_effect = lambda X: np.tile(
        [0.9, 0.1], (X.shape[0], 1)
    )

    # Wylaczamy lifespan lub nadpisujemy model po starcie, aby nie pobierac artefaktow
    with TestClient(app) as test_client:
        app.state.model = mock_model
        yield test_client
    app.state.model = None


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_zaladowany": True}


def test_model_info(client):
    mock_version = MagicMock(version="3", run_id="test_run_id")
    mock_run = MagicMock()
    mock_run.data.metrics = {"accuracy": 0.96}

    with patch("src.api.MlflowClient") as mock_client_cls:
        instance = mock_client_cls.return_value
        instance.get_model_version_by_alias.return_value = mock_version
        instance.get_run.return_value = mock_run

        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["wersja"] == "3"
        assert "accuracy" in data["metryki"]


def test_predict(client):
    cechy = [0.5] * LICZBA_CECH
    response = client.post("/predict", json={"cechy": cechy})
    assert response.status_code == 200
    data = response.json()
    assert data["klasa"] == 0
    assert 0.0 <= data["prawdopodobienstwo"] <= 1.0


def test_predict_blad_walidacji_422(client):
    response = client.post("/predict", json={"cechy": [1.0, 2.0]})
    assert response.status_code == 422


def test_predict_batch(client):
    obserwacje = [{"cechy": [0.5] * LICZBA_CECH} for _ in range(3)]
    response = client.post("/predict-batch", json={"obserwacje": obserwacje})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predykcje"]) == 3
    for pred in data["predykcje"]:
        assert pred["klasa"] == 0
