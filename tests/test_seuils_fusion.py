"""Tests automatiques (lancés par GitHub Actions, étape 9) : python -m pytest"""
import pandas as pd

from src.cleaning import nettoyer
from src.models.fusion import fusionner
from src.models.seuils import detecter_mesures


def _df(temps, ecart=0.02):
    n = len(temps)
    return pd.DataFrame({
        "vehicle_id": [1] * n,
        "time": pd.date_range("2020-01-01", periods=n, freq="8s"),
        "current": [-50.0] * n,
        "cell_v_max": [3.9 + ecart] * n,
        "cell_v_min": [3.9] * n,
        "temp_max": temps,
        "temp_min": [t - 2 for t in temps],
    })


def test_surchauffe_confirmee_apres_3_mesures():
    df = detecter_mesures(_df([40, 46, 46, 46]))
    assert df["surchauffe_confirmee"].tolist() == [False, False, False, True]


def test_pic_isole_non_confirme():
    df = detecter_mesures(_df([40, 46, 40, 40]))
    assert not df["surchauffe_confirmee"].any()


def test_cellule_defaillante():
    df = detecter_mesures(_df([30, 30], ecart=0.15))
    assert df["cellule_defaillante"].all()


def test_nettoyage_retire_erreurs_capteur():
    df = _df([30, 30, 30])
    df.loc[1, "cell_v_min"] = 0.0
    assert len(nettoyer(df)) == 2


def test_fusion_niveaux():
    s = pd.DataFrame({"vote_seuils": [0, 1, 1], "vote_regression": [0, 1, 1],
                      "vote_alerte_precoce": [0, 0, 1], "vote_isolation_forest": [1, 0, 0]})
    assert fusionner(s)["niveau_alerte"].tolist() == ["normal", "à surveiller", "critique"]
