"""Chemins et paramètres du projet (relatifs à la racine, aucun chemin absolu)."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # #1.csv ... #20.csv (données brutes)
PROCESSED_DIR = DATA_DIR / "processed"  # fichiers fusionnés / nettoyés / sessions
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
MODELS_DIR = ROOT_DIR / "models"        # modèles entraînés (non versionnés)

DATASET_PARQUET = PROCESSED_DIR / "dataset_batterie.parquet"     # étape 2
CLEAN_PARQUET = PROCESSED_DIR / "dataset_nettoye.parquet"        # étape 3
SESSIONS_PARQUET = PROCESSED_DIR / "sessions_features.parquet"   # étape 5
RESULTATS_PARQUET = PROCESSED_DIR / "resultats_sessions.parquet" # étape 6 (votes + niveau)

# Noms de colonnes courts (ordre des CSV bruts, après vehicle_id)
COLUMNS = ["vehicle_id", "time", "soc", "pack_v", "current", "cell_v_max", "cell_v_min",
           "temp_max", "temp_min", "energy", "capacity"]

# ---------- Seuils (rapport §1.2.3 et §5.2) ----------
TEMP_ALERTE = 45          # °C : surchauffe
TEMP_CRITIQUE = 48        # °C : 50 °C ne se déclencherait jamais (max observé 49 °C)
ECART_TENSION_MAX = 0.1   # V : cellule défaillante
FENETRE_CONFIRMATION = 3  # mesures consécutives pour confirmer une anomalie

# ---------- Sessions de charge (rapport §3.2) ----------
SESSION_GAP_S = 600       # nouvelle session après une coupure > 10 min
ALERTE_PRECOCE_MIN = 15   # le modèle 2 ne voit que les 15 premières minutes

# ---------- Évaluation (rapport §3.7) ----------
# Séparation par véhicule : 16 véhicules d'entraînement, 4 de test jamais vus
TEST_VEHICLES = [17, 18, 19, 20]   # à valider en équipe
RANDOM_STATE = 42

# ---------- Fusion par vote (rapport §3.6) ----------
# 0-1 vote : normal · 2 votes : à surveiller · 3-4 votes : critique
NIVEAUX_ALERTE = {0: "normal", 1: "normal", 2: "à surveiller", 3: "critique", 4: "critique"}
