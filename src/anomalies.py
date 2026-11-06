"""Nettoyage des erreurs capteur et détection d'anomalies batterie."""
import pandas as pd

from src.config import (ANOMALIES_PARQUET, ECART_TENSION_MAX, FENETRE_CONFIRMATION,
                        TEMP_ALERTE, TEMP_CRITIQUE)
from src.data_loading import charger_dataset


def nettoyer(df):
    df = df.drop_duplicates(subset=["vehicle_id", "time"]).copy()  # doublons d'horodatage
    df.loc[df["cell_v_min"] <= 0, "cell_v_min"] = pd.NA              # tension 0 V = capteur
    df.loc[~df["temp_max"].between(-30, 80), "temp_max"] = pd.NA    # température impossible
    return df


def _confirme(df, col):
    """Anomalie confirmée = N mesures consécutives (évite le bruit ponctuel)."""
    n = FENETRE_CONFIRMATION
    return (df.groupby("vehicle_id")[col]
              .transform(lambda s: s.astype(int).rolling(n).sum() == n))


def detecter(df):
    df["ecart_tension"] = df["cell_v_max"] - df["cell_v_min"]
    df["surchauffe_alerte"] = df["temp_max"] > TEMP_ALERTE
    df["surchauffe_critique"] = df["temp_max"] > TEMP_CRITIQUE
    df["cellule_defaillante"] = df["ecart_tension"] > ECART_TENSION_MAX
    df["surchauffe_confirmee"] = _confirme(df, "surchauffe_alerte")
    df["cellule_confirmee"] = _confirme(df, "cellule_defaillante")
    return df


def resume_par_vehicule(df):
    return df.groupby("vehicle_id").agg(
        mesures=("time", "size"),
        temp_max=("temp_max", "max"),
        surchauffe_45=("surchauffe_alerte", "sum"),
        surchauffe_50=("surchauffe_critique", "sum"),
        surchauffe_conf=("surchauffe_confirmee", "sum"),
        ecart_v_max=("ecart_tension", "max"),
        ecart_v_moyen=("ecart_tension", "mean"),
        cellule_0_1V=("cellule_defaillante", "sum"),
        cellule_conf=("cellule_confirmee", "sum"),
    )


if __name__ == "__main__":
    df = charger_dataset()
    avant = len(df)
    df = detecter(nettoyer(df))
    print(f"Doublons supprimés : {avant - len(df):,}")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(resume_par_vehicule(df).round(3))

    df.to_parquet(ANOMALIES_PARQUET, index=False)
    print("Sauvegardé :", ANOMALIES_PARQUET)
