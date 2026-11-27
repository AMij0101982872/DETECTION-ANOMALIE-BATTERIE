import numpy as np
import pandas as pd

from src.config import TEST_VEHICLES
from src.models.alerte_precoce import FEATURES, entrainer, population, separer, vote_sessions


def _sessions(n_par_vehicule=150, seed=0):
    """Sessions synthétiques : une session finit > 45 °C quand elle est déjà chaude à 15 min."""
    rng = np.random.default_rng(seed)
    lignes = []
    for v in range(1, 21):
        for i in range(n_par_vehicule):
            temp_debut = rng.uniform(20, 42)
            temp_15 = temp_debut + rng.uniform(0, 2)
            lignes.append({
                "vehicle_id": v, "session_id": v * 1000 + i, "session_longue": True,
                "temp_debut": temp_debut, "temp_15min": temp_15,
                "vitesse_temp_15min": (temp_15 - temp_debut) / 15,
                "courant_moyen_15min": rng.uniform(-80, -20), "ecart_v_max_15min": 0.03,
                "soc_15min": rng.uniform(20, 80), "soc_debut": rng.uniform(10, 70), "mois": rng.integers(1, 13),
                "temp_max": temp_15 + (6 if temp_15 > 41 else 1),
            })
    df = pd.DataFrame(lignes)
    # Deux cas particuliers : une session courte et une déjà en surchauffe à 15 min
    df.loc[0, "session_longue"] = False
    df.loc[1, ["temp_15min", "temp_max"]] = [46, 47]
    return df


def test_population_exclut_courtes_et_deja_en_surchauffe():
    df = _sessions()
    pop = population(df)
    assert len(pop) == len(df) - 2
    assert pop["temp_15min"].max() <= 45
    assert set(pop["label_surchauffe"]) == {0, 1}


def test_separation_par_vehicule():
    train, test = separer(population(_sessions()))
    assert set(test["vehicle_id"]) == set(TEST_VEHICLES)
    assert not set(train["vehicle_id"]) & set(TEST_VEHICLES)


def test_entrainement_et_votes():
    df = _sessions()
    resultats, modele, seuil = entrainer(df, journaliser=False, verbose=False)
    assert set(resultats["approche"]) == {"regle_temp_15min", "regression_logistique", "random_forest", "xgboost"}
    assert resultats["meilleur"].sum() == 1
    assert resultats.loc[resultats.meilleur, "rappel_test"].iloc[0] > 0.8

    votes = vote_sessions(df, modele, seuil)
    assert len(votes) == len(df)
    assert votes.loc[0, "vote_alerte_precoce_surchauffe"] == 0 and not votes.loc[0, "ap_evaluable"]   # courte
    assert votes.loc[1, "vote_alerte_precoce_surchauffe"] == 1                                         # déjà > 45 °C
    assert set(FEATURES).isdisjoint(votes.columns)
