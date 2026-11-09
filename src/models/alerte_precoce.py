"""Modèle 2 — Alerte précoce (supervisé, corrigé = seuils) : « ça va arriver ».

À FAIRE (membre 3) :
- étiquette : la session finit-elle au-delà de 45 °C (ou 0,1 V) ? -> fournie par models/seuils.py
- entrée : UNIQUEMENT les features des 15 premières minutes (sinon raisonnement circulaire, §3.4)
- algorithmes : Random Forest, XGBoost en classification
- métriques : précision, rappel, F1 — le rappel est prioritaire
- séparation par véhicule (config.TEST_VEHICLES) ; classes déséquilibrées -> class_weight
"""


def entrainer(sessions):
    raise NotImplementedError


def vote_sessions(sessions, modele):
    """1 si la probabilité d'anomalie dépasse le seuil choisi, sinon 0."""
    raise NotImplementedError
