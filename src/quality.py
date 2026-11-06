"""Rapport de qualité des données par véhicule."""
import pandas as pd

from src.data_loading import charger_dataset


def rapport_qualite(df):
    dt_s = df.groupby("vehicle_id")["time"].diff().dt.total_seconds()
    g = df.assign(dt_s=dt_s).groupby("vehicle_id")

    rapport = pd.DataFrame({
        "debut": g["time"].min(),
        "fin": g["time"].max(),
        "pas_median_s": g["dt_s"].median(),
        "trous_10min": g["dt_s"].apply(lambda s: (s > 600).sum()),
        "doublons": g["dt_s"].apply(lambda s: (s == 0).sum()),
        "cell_v_0": g["cell_v_min"].apply(lambda s: (s == 0).sum()),
        "courant_-400": g["current"].apply(lambda s: (s <= -400).sum()),
        "manquants": df.isna().groupby(df["vehicle_id"]).sum().sum(axis=1),
    })
    rapport["duree_jours"] = (rapport["fin"] - rapport["debut"]).dt.days
    return rapport


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(rapport_qualite(charger_dataset()))
