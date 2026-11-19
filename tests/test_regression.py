import numpy as np
import pandas as pd

from src.config import TEST_VEHICLES
from src.models.regression import CIBLES, FEATURES, entrainer, separer, vote_sessions


def _sessions(n_par_vehicule=60, seed=0):
    """Sessions synthétiques : temp_max dépend du courant et de la température de départ."""
    rng = np.random.default_rng(seed)
    lignes = []
    for v in range(1, 21):
        for i in range(n_par_vehicule):
            courant = rng.uniform(-80, -20)
            temp_debut = rng.uniform(15, 40)
            lignes.append({
                "vehicle_id": v, "session_id": v * 1000 + i,
                "courant_moyen": courant, "soc_debut": rng.uniform(10, 80),
                "temp_debut": temp_debut, "mois": rng.integers(1, 13),
                "duree_min": rng.uniform(20, 120),
                "temp_max": temp_debut + 0.1 * -courant + rng.normal(0, 0.5),
                "ecart_v_max": 0.04 + rng.normal(0, 0.002),
            })
    return pd.DataFrame(lignes)


def test_separation_par_vehicule():
    train, test = separer(_sessions())
    assert set(test["vehicle_id"]) == set(TEST_VEHICLES)
    assert not set(train["vehicle_id"]) & set(TEST_VEHICLES)


def test_entrainement_6_modeles_et_un_meilleur_par_cible():
    resultats, modeles, seuils = entrainer(_sessions(), journaliser=False, verbose=False)
    assert len(resultats) == 6
    assert resultats.groupby("anomalie")["meilleur"].sum().eq(1).all()
    assert set(modeles) == set(seuils) == set(CIBLES)
    # La relation est linéaire et forte : le R² test de temp_max doit être élevé
    assert resultats.query("cible == 'temp_max'")["r2_test"].max() > 0.9


def test_session_plus_chaude_que_prevu_vote_anomalie():
    sessions = _sessions()
    _, modeles, seuils = entrainer(sessions, journaliser=False, verbose=False)
    anormale = sessions.iloc[[0]].copy()
    anormale["temp_max"] += 10                      # 10 °C de plus que ce que prévoient les conditions
    votes = vote_sessions(pd.concat([sessions.iloc[[1]], anormale]), modeles, seuils)
    assert votes["vote_regression_surchauffe"].tolist() == [0, 1]
    assert votes["vote_regression"].iloc[1] == 1
    assert {f"vote_regression_{a}" for a in CIBLES} <= set(votes.columns)
    assert set(FEATURES).isdisjoint(votes.columns)
