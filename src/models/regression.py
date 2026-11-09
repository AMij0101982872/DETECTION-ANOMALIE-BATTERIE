"""Modèle 1 — Régression (supervisé, corrigé = mesure réelle) : « elle chauffe plus que prévu ».

À FAIRE (membre 3) :
- entrée : contexte de la session (courant, SOC, température de départ, mois)
- cibles : temp_max de la session (anomalie 1) et écart de tension max (anomalie 2)
- algorithmes : régression linéaire (référence), Random Forest, XGBoost
- anomalie = écart réel − prévu trop grand (ex. 25 °C prévus, 37 °C réels)
- métriques : MAE, RMSE, R² ; séparation par véhicule (config.TEST_VEHICLES)
- chaque entraînement enregistré dans MLflow
"""


def entrainer(sessions):
    raise NotImplementedError


def vote_sessions(sessions, modele):
    """1 si le résidu (réel − prévu) dépasse le seuil choisi, sinon 0."""
    raise NotImplementedError
