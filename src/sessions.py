"""Étape 5 — Découpage en sessions de charge et calcul des features (rapport §3.2).

À FAIRE (membre 1) :
- une nouvelle session commence après une coupure > SESSION_GAP_S (10 min), par véhicule
- ~30 000 sessions attendues
- features par session : durée, SOC début/fin, courant moyen et max, température de départ
  et max, vitesse de montée en température, écart de tension moyen et max, mois
- features des 15 premières minutes (ALERTE_PRECOCE_MIN) pour le modèle 2
- sortie : SESSIONS_PARQUET (une ligne par session, colonnes vehicle_id, session_id, features)
"""
from src.config import SESSION_GAP_S, ALERTE_PRECOCE_MIN, SESSIONS_PARQUET  # noqa: F401


def ajouter_session_id(df):
    """Ajoute une colonne session_id (coupure > 10 min ou changement de véhicule)."""
    raise NotImplementedError


def features_par_session(df):
    """Une ligne par session avec les features listées ci-dessus."""
    raise NotImplementedError


if __name__ == "__main__":
    from src.cleaning import charger_nettoye
    sessions = features_par_session(ajouter_session_id(charger_nettoye()))
    sessions.to_parquet(SESSIONS_PARQUET, index=False)
    print("Sauvegardé :", SESSIONS_PARQUET, sessions.shape)
