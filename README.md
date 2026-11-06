# Détection d'anomalies batterie – véhicules électriques

Analyse des données de charge de 20 véhicules électriques (juillet 2019 → novembre 2021, ~16,1 millions de mesures, une mesure toutes les 8 à 10 s) afin de détecter des anomalies batterie : **surchauffe** et **déséquilibre de tension entre cellules**.

## Architecture

```
Dataset_automobile/
├── data/
│   ├── raw/                 # #1.csv … #20.csv : un fichier brut par véhicule (non versionné)
│   └── processed/           # dataset_batterie.parquet, dataset_anomalies.parquet (non versionné)
├── notebooks/
│   └── 01_exploration_anomalies.ipynb   # exploration, qualité, détection
├── src/
│   ├── config.py            # chemins relatifs + seuils de détection
│   ├── data_loading.py      # fusion des CSV bruts -> parquet
│   ├── quality.py           # rapport qualité par véhicule
│   └── anomalies.py         # nettoyage + détection + résumé par véhicule
├── reports/figures/         # graphiques exportés
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Placez ensuite les fichiers `#1.csv` à `#20.csv` dans `data/raw/`.

## Utilisation

Depuis la racine du projet :

```bash
python -m src.data_loading   # 1. fusion des 20 CSV -> data/processed/dataset_batterie.parquet
python -m src.quality        # 2. rapport qualité par véhicule
python -m src.anomalies      # 3. détection -> data/processed/dataset_anomalies.parquet
```

## Données

| Colonne      | Description                     |
|--------------|---------------------------------|
| `vehicle_id` | numéro du véhicule (1 à 20)     |
| `time`       | horodatage de la mesure         |
| `soc`        | état de charge (%)              |
| `pack_v`     | tension du pack (V)             |
| `current`    | courant de charge (A)           |
| `cell_v_max` / `cell_v_min` | tension cellule max / min (V) |
| `temp_max` / `temp_min`     | température max / min (°C)    |
| `energy`     | énergie disponible (kWh)        |
| `capacity`   | capacité disponible (Ah)        |

## Règles de détection (modifiables dans `src/config.py`)

- **Surchauffe** : alerte si `temp_max` > 45 °C, critique si > 50 °C
- **Cellule défaillante** : écart `cell_v_max - cell_v_min` > 0,1 V
- **Anomalie confirmée** : condition vraie sur 3 mesures consécutives (filtre le bruit capteur)
- **Nettoyage** : suppression des doublons d'horodatage ; tension cellule à 0 V et températures hors [-30, 80] °C traitées comme erreurs capteur
