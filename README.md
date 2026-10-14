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