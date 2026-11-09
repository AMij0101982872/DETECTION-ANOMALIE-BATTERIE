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


def charger_nettoye():
    return pd.read_parquet(CLEAN_PARQUET)


if __name__ == "__main__":
    df = nettoyer(charger_dataset())
    df.to_parquet(CLEAN_PARQUET, index=False)
    print("Sauvegardé :", CLEAN_PARQUET, df.shape)
