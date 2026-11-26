# Etap 1: budowanie, instalacja zaleznosci do wydzielonego katalogu
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt
# Etap 2: finalny obraz uruchomieniowy
FROM python:3.12-slim
WORKDIR /app
# Przeniesienie gotowych pakietow z etapu budowania
COPY --from=builder /install /usr/local
# Utworzenie nieuprzywilejowanego uzytkownika
RUN useradd --create-home --shell /bin/bash appuser
# Konfiguracja i kod aplikacji z prawami dla appuser
COPY --chown=appuser:appuser config.yaml .
COPY --chown=appuser:appuser src/ ./src/

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]