"""Chemins et paramètres du projet (relatifs à la racine, aucun chemin absolu)."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # #1.csv ... #20.csv (données brutes)
PROCESSED_DIR = DATA_DIR / "processed"  # fichiers fusionnés / enrichis
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

DATASET_PARQUET = PROCESSED_DIR / "dataset_batterie.parquet"
ANOMALIES_PARQUET = PROCESSED_DIR / "dataset_anomalies.parquet"

# Noms de colonnes courts (ordre des CSV bruts, après vehicle_id)
COLUMNS = ["vehicle_id", "time", "soc", "pack_v", "current", "cell_v_max", "cell_v_min",
           "temp_max", "temp_min", "energy", "capacity"]

# Seuils de détection
TEMP_ALERTE = 45          # °C
TEMP_CRITIQUE = 50        # °C
ECART_TENSION_MAX = 0.1   # V entre cellule max et min
FENETRE_CONFIRMATION = 3  # mesures consécutives pour confirmer une anomalie
