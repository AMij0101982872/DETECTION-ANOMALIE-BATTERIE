"""Étape 5 — Sessions de charge et features par session."""
import argparse

import pandas as pd

from src.cleaning import charger_nettoye
from src.config import ALERTE_PRECOCE_MIN, SESSION_GAP_S, SESSIONS_PARQUET

# Choix documenté : une session de moins de 10 mesures (~1,5 min à 8 s/mesure)
# est trop courte pour calculer des features fiables (pente, moyennes).
MIN_MESURES = 10

COLONNES_SORTIE = [
    # Identité
    "vehicle_id", "session_id", "debut", "fin", "mois", "nb_mesures",
    # Session complète
    "duree_min", "soc_debut", "soc_fin", "courant_moyen", "courant_max",
    "temp_debut", "temp_max", "vitesse_temp", "ecart_v_moyen", "ecart_v_max",
    # 15 premières minutes (modèle 2)
    "temp_15min", "vitesse_temp_15min", "courant_moyen_15min",
    "ecart_v_max_15min", "soc_15min", "session_longue",
]


def ajouter_session_id(df):
    """Nouvelle session si le véhicule change ou si > SESSION_GAP_S entre deux mesures."""
    df = df.sort_values(["vehicle_id", "time"]).reset_index(drop=True)
    changement_vehicule = df["vehicle_id"].ne(df["vehicle_id"].shift())
    coupure = df["time"].diff().dt.total_seconds() > SESSION_GAP_S
    df["session_id"] = (changement_vehicule | coupure).cumsum()
    return df


def filtrer_sessions_courtes(df, min_mesures=MIN_MESURES):
    """Écarte les sessions de moins de `min_mesures` mesures."""
    taille = df.groupby("session_id")["time"].transform("size")
    return df[taille >= min_mesures].copy()


def features_par_session(df):
    """Une ligne par session. `df` doit contenir la colonne session_id."""
    df = df[["vehicle_id", "session_id", "time", "soc", "current",
    "temp_max", "cell_v_max", "cell_v_min"]].copy()
    df["ecart_v"] = df["cell_v_max"] - df["cell_v_min"]
    df["courant_abs"] = df["current"].abs()

    g = df.groupby("session_id", sort=True)
    feat = g.agg(
        vehicle_id=("vehicle_id", "first"),
        debut=("time", "first"),
        fin=("time", "last"),
        nb_mesures=("time", "size"),
        soc_debut=("soc", "first"),
        soc_fin=("soc", "last"),
        courant_moyen=("current", "mean"),
        courant_max=("courant_abs", "max"),       # valeur absolue
        temp_debut=("temp_max", "first"),
        temp_max=("temp_max", "max"),
        ecart_v_moyen=("ecart_v", "mean"),
        ecart_v_max=("ecart_v", "max"),
    )
    feat["mois"] = feat["debut"].dt.month
    feat["duree_min"] = (feat["fin"] - feat["debut"]).dt.total_seconds() / 60
    duree = feat["duree_min"].where(feat["duree_min"] > 0)
    feat["vitesse_temp"] = (feat["temp_max"] - feat["temp_debut"]) / duree   # °C/min

    # --- 15 premières minutes (modèle 2) ---
    debut_session = g["time"].transform("first")
    fenetre = df[(df["time"] - debut_session) <= pd.Timedelta(minutes=ALERTE_PRECOCE_MIN)]
    f15 = fenetre.groupby("session_id").agg(
        temp_15min=("temp_max", "max"),
        courant_moyen_15min=("current", "mean"),
        ecart_v_max_15min=("ecart_v", "max"),
        soc_15min=("soc", "last"),
        fin_15=("time", "last"),
    )
    feat = feat.join(f15)
    duree15 = ((feat["fin_15"] - feat["debut"]).dt.total_seconds() / 60)
    feat["vitesse_temp_15min"] = (feat["temp_15min"] - feat["temp_debut"]) / duree15.where(duree15 > 0)
    feat["session_longue"] = feat["duree_min"] > ALERTE_PRECOCE_MIN

    return feat.reset_index()[COLONNES_SORTIE]


def construire_sessions(df):
    n_mesures = len(df)
    df = ajouter_session_id(df)
    n_avant = df["session_id"].nunique()
    df = filtrer_sessions_courtes(df)
    n_apres = df["session_id"].nunique()
    print(f"Sessions détectées : {n_avant:,} ; gardées (>= {MIN_MESURES} mesures) : "
    f"{n_apres:,} ; écartées : {n_avant - n_apres:,}")
    print(f"Mesures : {n_mesures:,} -> {len(df):,}")
    return features_par_session(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicules", type=int, nargs="*",
                        help="Limiter à certains véhicules, ex. --vehicules 1 2")
    args = parser.parse_args()

    donnees = charger_nettoye()
    if args.vehicules:
        donnees = donnees[donnees["vehicle_id"].isin(args.vehicules)]
        print(f"Version partielle : véhicules {args.vehicules}")

    sessions = construire_sessions(donnees)
    sessions.to_parquet(SESSIONS_PARQUET, index=False)
    print("Sauvegardé :", SESSIONS_PARQUET, sessions.shape)
    print(sessions.describe().T[["count", "mean", "min", "max"]])