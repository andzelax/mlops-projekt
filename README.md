# MLOps Projekt

**Autor:** andzelax  
**Projekt:** Infrastruktura i architektura chmurowa dla systemów sztucznej inteligencji 

---

## 1. Porównanie narzędzi do zarządzania środowiskami i wersjonowania danych

| Narzędzie | Zmierzony czas instalacji pakietów | Plik definicji zależności | Sposób odtworzenia środowiska (1 komenda) | Kiedy stosować / Rekomendacja |
| :--- | :--- | :--- | :--- | :--- |
| **Venv** | **26m 41.355s** *(standardowy instalator pip)* | `requirements.txt` | `pip install -r requirements.txt` | Standardowe, małe projekty Pythona bez zewnętrznych bibliotek systemowych i kompilatorów C/CUDA; gdy środowisko docelowe nie pozwala na instalację zewnętrznych binariów. |
| **UV** | **0m 6.984s** *(ponad 200x szybciej niż pip)* | `requirements.txt` / `pyproject.toml` | `uv pip install -r requirements.txt` | Nowoczesne projekty produkcyjne, potoki CI/CD i konteneryzacja Docker, gdzie kluczowa jest maksymalna prędkość budowania środowiska i rozwiązywania zależności. |
| **Conda** | ~1m 30s | `environment.yml` | `conda env create -f environment.yml` | Złożone projekty Data Science / Deep Learning wymagające specyficznych bibliotek niskopoziomowych spoza PyPI (CUDA, cuDNN, sterowniki GPU, zoptymalizowane biblioteki numeryczne MKL). |
| **DVC** | *N/A (wersjonowanie danych)* | `*.dvc`, `.dvc/config` | `dvc pull` | Wersjonowanie dużych zbiorów danych, wag modeli oraz artefaktów uczenia maszynowego. DVC przechowuje w repozytorium Git jedynie lekkie sumy kontrolne (hashe MD5), a właściwe dane synchronizuje z zewnętrznym magazynem (lokalnym, S3, GDrive itp.). |

---

---

## 3. Odtworzenie projektu z czystego repozytorium (Ćwiczenie samodzielne nr 3)

Poniżej zestawiono sekwencję komend niezbędnych do pełnego odtworzenia środowiska oraz danych z poziomu świeżej maszyny / nowego katalogu:

```bash
# 1. Klonowanie repozytorium do nowego katalogu
cd ~
git clone ~/mlops-projekt test-odtworzenia
cd test-odtworzenia

# 2. Utworzenie wyizolowanego srodowiska wirtualnego (z uzyciem interpretera systemowego)
uv venv --python /usr/bin/python3

# 3. Aktywacja srodowiska
source .venv/bin/activate

# 4. Instalacja zaleznosci z pliku requirements.txt
uv pip install -r requirements.txt

# 5. Odtworzenie zbiorow danych z magazynu DVC na podstawie plikow wskaznikowych
dvc pull

# 6. Weryfikacja poprawnosci odtworzenia danych i statusu
ls -lh data/
dvc status

```
## Laboratorium nr 2

================================================================================
Ćwiczenie samodzielne nr 2: Porównanie z modelem bazowym
================================================================================

1. Tabela wyników:

+----------------------------+-----------------------+-------------------------+
| Model                      | Dokładność CV (5-fold)| Dokładność na testowym  |
+----------------------------+-----------------------+-------------------------+
| Model bazowy (Baseline)    | 0.9538 (± 0.0235)     | 0.9561                  |
| Model strojony (Optuna)    | 0.9582                | 0.9561                  |
+----------------------------+-----------------------+-------------------------+

Najlepsza znaleziona konfiguracja hiperparametrów:
- n_estimators: 366
- max_depth: 15
- min_samples_split: 10
- min_samples_leaf: 3
- criterion: 'gini'


2. Komentarz i odpowiedzi na pytania:

- Czy strojenie przyniosło poprawę?
Na etapie 5-krotnej walidacji krzyżowej proces strojenia przyniósł niewielką poprawę średniej dokładności (wzrost z poziomu 0.9538 do 0.9582). Na wydzielonym zbiorze testowym uzyskana dokładność modelu strojonego była jednak identyczna z modelem bazowym i wyniosła 0.9561. Świadczy to o tym, że domyślna konfiguracja lasu losowego w bibliotece scikit-learn stanowi już bardzo mocny punkt odniesienia dla stosunkowo prostego i niewielkiego zbioru Breast Cancer Wisconsin, a optymalizacja nie pozwoliła na dalszy zauważalny zysk na zbiorze testowym.

- Czy relacja między wynikiem walidacyjnym a testowym wskazuje na nadmierne dopasowanie doboru hiperparametrów do walidacji?
Nie, uzyskane wyniki nie wykazują objawów nadmiernego dopasowania (overfittingu) doboru hiperparametrów do podziału walidacyjnego. Wynik osiągnięty na zbiorze testowym (0.9561) jest niemal identyczny z wynikiem walidacji krzyżowej (0.9582) i w pełni mieści się w granicach błędu wyznaczonego przez odchylenie standardowe walidacji modelu bazowego (0.0235). Model zachował stabilną zdolność generalizacji na nowych, niewidzianych danych.

================================================================================
Ćwiczenie samodzielne nr 3: Interpretacja wizualizacji
================================================================================

1. Historia optymalizacji (Optimization History Plot):
Wykres przedstawia wartości funkcji celu uzyskane w kolejnych 50 próbach (niebieskie punkty) oraz krzywą najlepszego dotychczasowego wyniku (czerwona linia). Na wykresie widać, że optymalną wartość dokładności walidacji krzyżowej (0.9582) osiągnięto już w 11. próbie, po czym linia najlepszego wyniku uległa całkowitemu wypłaszczeniu aż do końca eksperymentu. Wniosek praktyczny jest taki, że budżet obliczeniowy rzędu 50 prób był w zupełności wystarczający dla tej przestrzeni, a dalsze zwiększanie liczby prób bez modyfikacji zakresów parametrów nie przyniosłoby już zauważalnej poprawy.

2. Ważność hiperparametrów (Hyperparameter Importances Plot):
Wykres ilustruje względny wpływ poszczególnych hiperparametrów na wariancję wartości funkcji celu, oszacowany przez bibliotekę Optuna. Dominujący wpływ na jakość klasyfikacji miał parametr n_estimators (około 59%), a w dalszej kolejności parametry kontrolujące podziały i złożoność drzew: min_samples_split (15%) oraz min_samples_leaf (13%), podczas gdy criterion miał wpływ marginalny (4%). W praktyce oznacza to, że w kolejnych etapach strojenia można pominąć poszukiwanie kryterium podziału (pozostawiając domyślne "gini"), a budżet obliczeniowy skoncentrować na doborze odpowiedniej liczby estymatorów oraz regularyzacji liści drzewa.

## Laboratorium 4: Pre-commit hook demonstration
text
(mlops) an_ga@Laptop:~/mlops-projekt$ git commit -m "test: demonstracja pre-commit"
[INFO] Installing environment for [https://github.com/astral-sh/ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit).
[INFO] Once installed this environment will be reused.
ruff.....................................................................Failed
- hook id: ruff
- files were modified by this hook

Found 1 error (1 fixed, 0 remaining).

ruff-format..............................................................Passed

## Laboratorium 6
## Rozwiązywanie problemów: sieć Docker w WSL2

W środowisku WSL2 + Docker Desktop bezpośrednie użycie flagi --add-host=host.docker.internal:host-gateway może powodować problemy z trasowaniem ruchu między kontenerami, ponieważ nazwa host.docker.internal jest już zarządzana przez Docker Desktopa i wskazuje na hosta Windows. Dodatkowo serwer MLflow 3.x domyślnie odrzuca żądania z nieznanych nazw hostów błędem 403 Invalid Host header (wbudowana ochrona przed atakami DNS rebinding).

Rozwiązanie

Aby zapewnić stabilną komunikację między kontenerem API a serwerem MLflow w WSL2:

Utwórz dedykowaną sieć Dockera:

docker network create mlops-net


Uruchom serwer MLflow w tej sieci, dopisując nazwę kontenera (mlflow-server:*) do parametru --allowed-hosts:

docker run -d \
  --name mlflow-server \
  --network mlops-net \
  -p 5000:5000 \
  -v mlflow-data:/mlflow \
  ghcr.io/mlflow/mlflow:v3.14.0 \
  mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:////mlflow/mlflow.db \
  --artifacts-destination /mlflow/artifacts \
  --cors-allowed-origins "*" \
  --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*,mlflow-server:*"


Uruchom kontener API w tej samej sieci (--network mlops-net), przekazując adres kontenera MLflow przez zmienną środowiskową:

docker run -d \
  --name projekt-api \
  --network mlops-net \
  -p 8000:8000 \
  -e MLFLOW_URI=http://mlflow-server:5000 \
  -e PYTHONUNBUFFERED=1 \
  projekt-api:0.1


  REPOSITORY    TAG       IMAGE ID       CREATED          SIZE
projekt-api   0.2       860288509edb   44 seconds ago   1.25GB
projekt-api   0.1       e9b52db04da8   19 minutes ago   1.67GB

## Optymalizacja obrazu Docker (`multi-stage build`)

| Wersja obrazu | Charakterystyka | Rozmiar |
| :--- | :--- | :--- |
| `projekt-api:0.1` | Jednoetapowy `Dockerfile` (pozostałości po `pip` i cache w warstwach) | `1.67 GB` |
| `projekt-api:0.2` | Budowa wieloetapowa (`--no-cache-dir`, przeniesienie `/install`, `.dockerignore`, użytkownik `appuser`) | `1.25 GB` |

Przejście na budowę wieloetapową oraz wykluczenie zbędnych plików w `.dockerignore` zmniejszyło rozmiar obrazu o **420 MB** (~25%), a uruchomienie kontenera na użytkowniku `appuser` ogranicza potencjalne skutki przejęcia procesu aplikacji.