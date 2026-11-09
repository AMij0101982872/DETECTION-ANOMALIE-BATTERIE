"""Étape 8 — Dashboard Streamlit : streamlit run dashboard/app.py

À FAIRE (membre 2) :
- niveau d'alerte par session et par véhicule (lu depuis RESULTATS_PARQUET)
- classement des véhicules par score 0-100 (fusion.score_vehicules)
- historique des alertes, filtres par véhicule et par période
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.title("Supervision des batteries — détection d'anomalies")
st.info("Dashboard en construction (étape 8).")
