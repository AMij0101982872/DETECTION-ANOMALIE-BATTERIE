"""Fusion des CSV bruts (un fichier par véhicule) en un seul fichier parquet."""
import pandas as pd

from src.config import COLUMNS, DATASET_PARQUET, RAW_DIR


def lister_fichiers_bruts():
    """#1.csv, #2.csv, ..., #20.csv triés numériquement (et non #1, #10, #11, #2)."""
    return sorted(RAW_DIR.glob("#*.csv"), key=lambda f: int(f.stem.lstrip("#")))


def fusionner_csv(verbose=True):
    frames = []
    for f in lister_fichiers_bruts():
        d = pd.read_csv(f, index_col=0)  # supprime "Unnamed: 0"
        d.insert(0, "vehicle_id", int(f.stem.lstrip("#")))
        frames.append(d)
        if verbose:
            print(f"{f.name}: {len(d):,} lignes")

    df = pd.concat(frames, ignore_index=True)
    df.columns = COLUMNS
    df["time"] = pd.to_datetime(df["time"].astype(str), format="%Y%m%d%H%M%S")
    return df.sort_values(["vehicle_id", "time"]).reset_index(drop=True)


def charger_dataset():
    return pd.read_parquet(DATASET_PARQUET)


if __name__ == "__main__":
    df = fusionner_csv()
    DATASET_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PARQUET, index=False)
    print("Fusion OK :", df.shape, "->", DATASET_PARQUET)
