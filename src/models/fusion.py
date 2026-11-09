"""Fusion par vote (rapport §3.6) : 0-1 vote normal · 2 à surveiller · 3-4 critique."""
import pandas as pd

from src.config import NIVEAUX_ALERTE

COLONNES_VOTE = ["vote_seuils", "vote_regression", "vote_alerte_precoce", "vote_isolation_forest"]


def fusionner(sessions):
    """Attend les 4 colonnes de vote (0/1) et ajoute nb_votes et niveau_alerte."""
    sessions = sessions.copy()
    sessions["nb_votes"] = sessions[COLONNES_VOTE].sum(axis=1).astype(int)
    sessions["niveau_alerte"] = sessions["nb_votes"].map(NIVEAUX_ALERTE)
    return sessions


def score_vehicules(sessions):
    """Score global 0-100 par véhicule pour le classement du dashboard (à affiner)."""
    s = sessions.groupby("vehicle_id")["nb_votes"].mean() / len(COLONNES_VOTE) * 100
    return s.round(1).sort_values(ascending=False).rename("score")


if __name__ == "__main__":
    demo = pd.DataFrame({"vehicle_id": [1, 1, 2],
                         "vote_seuils": [0, 1, 1], "vote_regression": [0, 1, 1],
                         "vote_alerte_precoce": [0, 0, 1], "vote_isolation_forest": [1, 0, 1]})
    print(fusionner(demo))
