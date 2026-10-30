import numpy as np

from src.data import load_data, split_data
from src.evaluate import evaluate_model
from src.train import train_model


def test_load_data():
    """Test danych: wymiary, kompletnosc i liczba klas."""
    X, y = load_data()
    assert X.shape[0] == y.shape[0]
    assert not np.isnan(X).any()
    assert len(np.unique(y)) == 2


def test_ksztalt_predykcji():
    """Test kontraktu: ksztalt i dziedzina predykcji."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    predykcje = model.predict(X_test)
    assert predykcje.shape[0] == X_test.shape[0]
    assert set(np.unique(predykcje)).issubset({0, 1})


def test_prog_trafnosci():
    """Test walidacyjny: minimalna akceptowalna jakosc modelu."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy > 0.90


def test_split_data_proporcje():
    """Wlasny test 1: weryfikacja poprawnosci licznosci i proporcji test_size."""
    X, y = load_data()
    test_size = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    # Zgodnosc sumarycznej licznosci
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

    # Poprawnosc proporcji (z marginesem zaokraglenia do 1 obserwacji)
    oczekiwany_rozmiar_test = int(np.ceil(len(X) * test_size))
    assert abs(len(X_test) - oczekiwany_rozmiar_test) <= 1


def test_split_data_stratyfikacja():
    """Wlasny test 2: zachowanie proporcji klas po podziale stratyfikowanym."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    proporcja_oryginalna = float(np.mean(y == 1))
    proporcja_train = float(np.mean(y_train == 1))
    proporcja_test = float(np.mean(y_test == 1))

    # Roznice w udziale klas nie powinny przekraczac 2 punktow procentowych
    assert abs(proporcja_train - proporcja_oryginalna) < 0.02
    assert abs(proporcja_test - proporcja_oryginalna) < 0.02
