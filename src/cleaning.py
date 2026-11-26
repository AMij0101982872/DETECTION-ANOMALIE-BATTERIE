"""Étape 3 — Nettoyage : les erreurs capteur sont retirées, pas traitées comme des anomalies."""
import pandas as pd

from src.config import CLEAN_PARQUET
from src.data_loading import charger_dataset


def nettoyer(df):
    avant = len(df)
    df = df.drop_duplicates(subset=["vehicle_id", "time"]).copy()   # doublons d'horodatage
    df = df[(df["cell_v_min"] > 0) & (df["cell_v_max"] > 0)]         # tension cellule à 0 V
    df = df[df["current"] > -400]                                     # code d'erreur −400 A
    df = df[df["temp_max"].between(-30, 80) & df["temp_min"].between(-30, 80)]
    print(f"Lignes retirées au nettoyage : {avant - len(df):,}")
    return df.reset_index(drop=True)


def compter_erreurs(df):
    """Compte les lignes retirées par type d'erreur, dans l'ordre d'application.

    Une ligne qui cumule plusieurs erreurs n'est comptée qu'une fois (à la
    première étape qui la retire), donc la somme égale le total retiré.
    """
    stats = {}

    n = len(df)
    df = df.drop_duplicates(subset=["vehicle_id", "time"])
    stats["doublons (vehicle_id, time)"] = n - len(df)

    n = len(df)
    df = df[(df["cell_v_min"] > 0) & (df["cell_v_max"] > 0)]
    stats["tension cellule = 0 V"] = n - len(df)

    n = len(df)
    df = df[df["current"] > -400]
    stats["courant = -400 A"] = n - len(df)

    n = len(df)
    df = df[df["temp_max"].between(-30, 80) & df["temp_min"].between(-30, 80)]
    stats["température hors [-30, 80] °C"] = n - len(df)

    stats["TOTAL"] = sum(stats.values())
    return stats


def charger_nettoye():
    return pd.read_parquet(CLEAN_PARQUET)


if __name__ == "__main__":
    brut = charger_dataset()
    print("Détail par type d'erreur (pour le Table 4) :")
    for nom, nb in compter_erreurs(brut).items():
        print(f"  {nom:<32} {nb:>10,}")
    df = nettoyer(brut)
    df.to_parquet(CLEAN_PARQUET, index=False)
    print("Sauvegardé :", CLEAN_PARQUET, df.shape)