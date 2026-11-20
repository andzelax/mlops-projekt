from contextlib import asynccontextmanager

import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

MLFLOW_URI = "http://127.0.0.1:5000"
NAZWA_MODELU = "breast_cancer"
ALIAS_MODELU = "champion"
LICZBA_CECH = 30


class ZadaniePredykcji(BaseModel):
    cechy: list[float] = Field(
        ...,
        min_length=LICZBA_CECH,
        max_length=LICZBA_CECH,
        description="Wektor cech pojedynczej obserwacji",
    )


class OdpowiedzPredykcji(BaseModel):
    klasa: int = Field(description="Przewidziana etykieta klasy")
    prawdopodobienstwo: float = Field(
        ge=0.0,
        le=1.0,
        description="Prawdopodobienstwo przewidzianej klasy",
    )


class ModelInfoResponse(BaseModel):
    nazwa_modelu: str = Field(description="Nazwa zarejestrowanego modelu")
    wersja: str = Field(description="Numer wersji wskazywanej przez alias")
    metryki: dict[str, float] = Field(
        description="Metryki jakosci zapisane w uruchomieniu"
    )


class ZadaniePredykcjiBatch(BaseModel):
    obserwacje: list[ZadaniePredykcji] = Field(
        min_length=1,
        description="Lista wektorow cech do predykcji wsadowej",
    )


class OdpowiedzPredykcjiBatch(BaseModel):
    predykcje: list[OdpowiedzPredykcji] = Field(
        description="Lista przewidzianych klas i prawdopodobienstw",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ladowanie modelu przy starcie aplikacji i sprzatanie przy zamknieciu."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        app.state.model = mlflow.sklearn.load_model(
            f"models:/{NAZWA_MODELU}@{ALIAS_MODELU}"
        )
    except Exception as e:
        print(f"Nie udalo sie zaladowac modelu z MLflow: {e}")
        app.state.model = None

    yield
    app.state.model = None


app = FastAPI(
    title="API projektu semestralnego",
    description="Serwowanie modelu klasyfikacyjnego z rejestru MLflow",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    model_zaladowany = getattr(app.state, "model", None) is not None
    return {"status": "ok", "model_zaladowany": model_zaladowany}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    try:
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        wersja_modelu = client.get_model_version_by_alias(NAZWA_MODELU, ALIAS_MODELU)
        run = client.get_run(wersja_modelu.run_id)
        return ModelInfoResponse(
            nazwa_modelu=NAZWA_MODELU,
            wersja=str(wersja_modelu.version),
            metryki=run.data.metrics,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Blad podczas pobierania metadanych z MLflow: {e}",
        )


@app.post("/predict", response_model=OdpowiedzPredykcji)
def predict(zadanie: ZadaniePredykcji):
    obserwacja = np.array(zadanie.cechy).reshape(1, -1)
    klasa = int(app.state.model.predict(obserwacja)[0])
    prawdopodobienstwa = app.state.model.predict_proba(obserwacja)[0]
    return OdpowiedzPredykcji(
        klasa=klasa,
        prawdopodobienstwo=float(prawdopodobienstwa[klasa]),
    )


@app.post("/predict-batch", response_model=OdpowiedzPredykcjiBatch)
def predict_batch(zadanie: ZadaniePredykcjiBatch):
    if getattr(app.state, "model", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Model nie zostal prawidlowo zaladowany przy starcie serwera.",
        )

    macierz = np.array([item.cechy for item in zadanie.obserwacje])
    klasy = app.state.model.predict(macierz)
    prawdopodobienstwa = app.state.model.predict_proba(macierz)

    wyniki = [
        OdpowiedzPredykcji(
            klasa=int(klasa),
            prawdopodobienstwo=float(proba[klasa]),
        )
        for klasa, proba in zip(klasy, prawdopodobienstwa)
    ]
    return OdpowiedzPredykcjiBatch(predykcje=wyniki)
