import pandas as pd

from src.sessions import ajouter_session_id, features_par_session


def _mesures(vehicle_id, debut, n, pas_s=8):
    t0 = pd.Timestamp(debut)
    return pd.DataFrame({
        "vehicle_id": vehicle_id,
        "time": [t0 + pd.Timedelta(seconds=pas_s * i) for i in range(n)],
        "soc": [20.0 + i * 0.1 for i in range(n)],
        "current": -50.0,
        "cell_v_max": 3.9, "cell_v_min": 3.85,
        "temp_max": [30 + i // 50 for i in range(n)],
    })


def test_coupure_de_plus_de_10_min_cree_une_session():
    a = _mesures(1, "2019-07-26 20:00:00", 20)
    b = _mesures(1, "2019-07-26 21:00:00", 20)   # 1 h plus tard
    df = ajouter_session_id(pd.concat([a, b], ignore_index=True))
    assert df["session_id"].nunique() == 2


def test_changement_de_vehicule_cree_une_session():
    a = _mesures(1, "2019-07-26 20:00:00", 20)
    b = _mesures(2, "2019-07-26 20:03:00", 20)   # pas de coupure, autre véhicule
    df = ajouter_session_id(pd.concat([a, b], ignore_index=True))
    assert df["session_id"].nunique() == 2


def test_features_une_ligne_par_session():
    df = ajouter_session_id(_mesures(1, "2019-07-26 20:00:00", 300))   # 40 min
    f = features_par_session(df)
    assert len(f) == 1
    ligne = f.iloc[0]
    assert ligne["nb_mesures"] == 300
    assert ligne["courant_max"] == 50.0           # valeur absolue
    assert round(ligne["ecart_v_max"], 2) == 0.05
    assert bool(ligne["session_longue"]) is True