"""Tests du modèle 3 (Isolation Forest) sur des sessions synthétiques."""
import numpy as np
import pandas as pd
import pytest

from src.config import TEST_VEHICLES
from src.models import isolation_forest as iforest


def _sessions(n_par_vehicule=80, seed=0):
    rng = np.random.default_rng(seed)
    lignes = []
    for v in range(1, 21):
        n = n_par_vehicule
        temp_debut = rng.normal(25, 3, n)
        duree = rng.uniform(20, 120, n)
        temp_max = temp_debut + rng.normal(8, 1.5, n)
        lignes.append(pd.DataFrame({
            "vehicle_id": v,
            "duree_min": duree,
            "soc_debut": rng.uniform(20, 50, n),
            "soc_fin": rng.uniform(80, 100, n),
            "courant_moyen": rng.normal(-60, 5, n),
            "courant_max": rng.normal(80, 5, n),
            "temp_debut": temp_debut,
            "temp_max": temp_max,
            "vitesse_temp": (temp_max - temp_debut) / duree,
            "ecart_v_moyen": rng.normal(0.035, 0.002, n),
            "ecart_v_max": rng.normal(0.041, 0.003, n),
        }))
    return pd.concat(lignes, ignore_index=True)


@pytest.fixture(scope="module")
def donnees():
    sessions = _sessions()
    return sessions, iforest.entrainer(sessions, n_estimators=100)


def test_vote_binaire_et_contamination(donnees):
    sessions, modele = donnees
    votes = iforest.vote_sessions(sessions, modele)
    assert set(votes.unique()) <= {0, 1}
    train = ~sessions["vehicle_id"].isin(TEST_VEHICLES)
    assert votes[train].mean() == pytest.approx(iforest.CONTAMINATION, abs=0.01)


def test_score_plus_grand_pour_session_atypique(donnees):
    sessions, modele = donnees
    normale = sessions.iloc[[0]]
    bizarre = normale.assign(temp_max=normale["temp_max"] + 20, ecart_v_max=0.15)
    assert iforest.score_sessions(bizarre, modele).iloc[0] > iforest.score_sessions(normale, modele).iloc[0]


def test_injection_ne_modifie_pas_l_original(donnees):
    sessions, _ = donnees
    avant = sessions.copy()
    injectees = iforest.injecter_anomalies(sessions, "cellule", n=10)
    pd.testing.assert_frame_equal(sessions, avant)
    origine = sessions.loc[injectees.index, "ecart_v_max"]
    assert np.allclose(injectees["ecart_v_max"] - origine, 0.060)


def test_injection_evaluee(donnees):
    sessions, modele = donnees
    resultats = iforest.evaluer_injection(sessions, modele, n=100)
    for type_anomalie in iforest.INJECTION:
        assert 0 <= resultats[f"rappel_{type_anomalie}"] <= 1
        # le score doit mieux classer les sessions modifiées que le hasard
        assert resultats[f"auroc_{type_anomalie}"] > 0.7


def test_valeurs_manquantes_et_infinies_acceptees(donnees):
    sessions, modele = donnees
    s = sessions.head(3).copy()
    s.loc[s.index[0], "vitesse_temp"] = np.inf
    s.loc[s.index[1], "soc_fin"] = np.nan
    assert len(iforest.vote_sessions(s, modele)) == 3
