"""Tests du modèle 3 (Isolation Forest unilatéral) sur des sessions synthétiques."""
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
    sessions = pd.concat(lignes, ignore_index=True)
    sessions["session_id"] = np.arange(1, len(sessions) + 1)
    return sessions


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


def test_ecretage_aucun_vote_pour_valeurs_basses(donnees):
    # Écart quasi nul et température très basse : bizarre, mais sans danger → pas de vote
    sessions, modele = donnees
    basse = sessions.iloc[[0]].assign(temp_max=0.0, vitesse_temp=0.0, ecart_v_max=0.0)
    assert iforest.vote_sessions(basse, modele).iloc[0] == 0


def test_session_courte_non_evaluable(donnees):
    sessions, modele = donnees
    courte = sessions.iloc[[0]].assign(duree_min=2.0, temp_max=80.0, ecart_v_max=0.2)
    assert iforest.vote_sessions(courte, modele).iloc[0] == 0
    table = iforest.table_votes(courte, modele)
    assert not table["if_evaluable"].iloc[0]


def test_table_votes_respecte_le_contrat(donnees):
    sessions, modele = donnees
    table = iforest.table_votes(sessions, modele)
    assert list(table.columns[:3]) == ["vehicle_id", "session_id", "vote_isolation_forest"]
    assert len(table) == len(sessions) and table["session_id"].is_unique
    assert set(table["vote_isolation_forest"].unique()) <= {0, 1}
    # une raison pour chaque vote, aucune pour les sessions normales
    assert table.loc[table["vote_isolation_forest"] == 1, "raison_isolation_forest"].notna().all()
    assert table.loc[table["vote_isolation_forest"] == 0, "raison_isolation_forest"].isna().all()


def test_explication_cellule(donnees):
    sessions, modele = donnees
    cellule = sessions.iloc[[0]].assign(ecart_v_max=0.15)
    assert iforest.expliquer(cellule, modele).iloc[0] == "cellule"


def test_injection_ne_modifie_pas_l_original(donnees):
    sessions, _ = donnees
    avant = sessions.copy()
    injectees = iforest.injecter_anomalies(sessions, "cellule", n=10)
    pd.testing.assert_frame_equal(sessions, avant)
    origine = sessions.loc[injectees.index]
    assert np.allclose(injectees["ecart_v_max"] - origine["ecart_v_max"], 0.060)
    assert np.allclose(injectees["ecart_v_moyen"] - origine["ecart_v_moyen"],
                       iforest.RATIO_ECART_MOYEN * 0.060)


def test_injection_evaluee(donnees):
    sessions, modele = donnees
    resultats = iforest.evaluer_injection(sessions, modele, n=100)
    assert 0 <= resultats["fausses_alarmes"] <= 1
    for type_anomalie in iforest.INJECTION:
        assert 0 <= resultats[f"rappel_{type_anomalie}"] <= 1
        # le score doit mieux classer les sessions modifiées que le hasard
        assert resultats[f"auroc_{type_anomalie}"] > 0.7


def test_valeurs_manquantes_et_infinies_acceptees(donnees):
    sessions, modele = donnees
    s = sessions.head(3).copy()
    s.loc[s.index[0], "vitesse_temp"] = np.inf
    s.loc[s.index[1], "temp_max"] = np.nan
    assert len(iforest.vote_sessions(s, modele)) == 3
    assert len(iforest.table_votes(s, modele)) == 3


def test_enregistrement_mlflow_et_rechargement(donnees, tmp_path, monkeypatch):
    # Le modèle (avec EcretageBas) se sauvegarde au format skops et se recharge à l'identique
    from src import mlops
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(mlops, "ARTIFACTS_DIR", tmp_path / "mlartifacts")
    sessions, modele = donnees
    iforest.enregistrer_mlflow(modele, sessions, {"rappel_surchauffe": 0.5})
    recharge = mlops.charger_modele("isolation_forest")
    pd.testing.assert_series_equal(iforest.score_sessions(sessions, recharge),
                                   iforest.score_sessions(sessions, modele))
