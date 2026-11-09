# Détection d'anomalies des batteries de véhicules électriques

Projet 3pH · PIC — Filière Big Data & IA, HESTIM (2026-2027)

**Problématique :** comment détecter, expliquer et anticiper automatiquement la **surchauffe** et la **cellule défaillante** d'une batterie de véhicule électrique, à partir de millions de mesures réelles et sans étiquettes de panne ?

Données : 16,1 millions de mesures de charge, 20 véhicules BAIC EU500, juillet 2019 → novembre 2021 (Deng et al., *Applied Energy*, 2023).

## Les 9 étapes du pipeline

| # | Étape | Statut | Code |
|---|---|---|---|
| 1 | Collecte des données | ✅ Réalisé | `data/raw/` |
| 2 | Fusion et stockage | ✅ Réalisé | `src/data_loading.py` |
| 3 | Contrôle qualité et nettoyage | 🟠 En cours | `src/quality.py`, `src/cleaning.py` |
| 4 | Analyse exploratoire | 🟠 En cours | `notebooks/` |
| 5 | Sessions de charge et features | ⬜ À faire | `src/sessions.py` |
| 6 | Modèles de machine learning | ⬜ À faire | `src/models/` |
| 7 | Sauvegarde et intégration (MLflow, Docker) | ⬜ À faire | `Dockerfile`, MLflow |
| 8 | Dashboard de supervision | ⬜ À faire | `dashboard/app.py` |
| 9 | Supervision et amélioration continue | ⬜ À faire | `tests/`, `.github/workflows/`, Evidently |

## Les 4 approches de détection + fusion par vote

| Approche | Type | Algorithmes | Fichier |
|---|---|---|---|
| Seuils — « ça dépasse maintenant » | Règles | 45 °C / 48 °C, 0,1 V | `src/models/seuils.py` |
| Modèle 1 : Régression — « elle chauffe plus que prévu » | Supervisé (mesures) | Linéaire, Random Forest, XGBoost | `src/models/regression.py` |
| Modèle 2 : Alerte précoce — « ça va arriver » | Supervisé (seuils) | Random Forest, XGBoost | `src/models/alerte_precoce.py` |
| Modèle 3 : Atypique — « cette charge est bizarre » | Non supervisé | Isolation Forest | `src/models/isolation_forest.py` |
| **Fusion** : 0-1 vote normal · 2 à surveiller · 3-4 critique | — | — | `src/models/fusion.py` |

## Répartition de l'équipe

👉 **Le détail des tâches de chacun est dans [TACHE.md](TACHE.md).**

| Membre | Branche | Rôle | Étapes | Fichiers |
|---|---|---|---|---|
| **1 — Kabre Yann Kevin** | `kevin` | Données | 2, 3, 5 | `data_loading.py`, `quality.py`, `cleaning.py`, `sessions.py` |
| **2 — Traore Mohamed** | `MOHAMED` | Analyse, seuils et dashboard | 4, 8 + seuils + fusion | `notebooks/`, `models/seuils.py`, `models/fusion.py`, `dashboard/` |
| **3 — Mobio Ivan Junior** | `IVANJR` | ML supervisé | 6 (modèles 1 et 2) | `models/regression.py`, `models/alerte_precoce.py` |
| **4 — Gatsoundou Junior Stevy** | `STEVY` | ML non supervisé et MLOps | 6 (modèle 3), 7, 9 | `models/isolation_forest.py`, MLflow, Docker, GitHub Actions, Evidently |

**Chemin critique :** l'étape 5 (sessions et features) bloque tous les modèles. Kevin doit la livrer en priorité ; en attendant, les autres préparent leur code sur un échantillon.

## Architecture

```
Dataset_automobile/
├── data/raw/              # #1.csv … #20.csv (versionnés via Git LFS)
├── data/processed/        # parquet fusionné, nettoyé, sessions (non versionnés)
├── notebooks/             # exploration et présentation des résultats
├── src/
│   ├── config.py          # chemins, seuils, paramètres
│   ├── data_loading.py    # étape 2
│   ├── quality.py         # étape 3 : rapport qualité
│   ├── cleaning.py        # étape 3 : nettoyage des erreurs capteur
│   ├── sessions.py        # étape 5
│   └── models/            # étape 6 : seuils, régression, alerte précoce, Isolation Forest, fusion
├── dashboard/app.py       # étape 8 (Streamlit)
├── tests/                 # étape 9 (pytest, lancé par GitHub Actions)
├── main.py                # pipeline complet en une commande
├── reports/figures/
└── requirements.txt
```

## Installation et utilisation

```bash
git lfs install                     # une seule fois, avant le clone : télécharge les CSV de data/raw/
pip install -r requirements.txt

python main.py                      # pipeline complet
python -m src.quality               # rapport qualité par véhicule
python -m src.models.seuils         # résumé des anomalies par seuils
python -m pytest                    # tests
streamlit run dashboard/app.py      # dashboard
```

## Règles d'évaluation (à respecter par tous)

- **Séparation par véhicule** : entraînement sur 16 véhicules, test sur 4 jamais vus (`config.TEST_VEHICLES`).
- **Anomalie confirmée** : au moins 3 mesures consécutives.
- **Modèle 2** : ne voit que les 15 premières minutes de la session (sinon raisonnement circulaire).
- **Erreurs capteur** : retirées au nettoyage, jamais comptées comme anomalies.

## Travail en équipe (Git)

Chaque membre travaille sur **sa** branche, jamais directement sur `main` :

```bash
git clone https://github.com/AMij0101982872/DETECTION-ANOMALIE-BATTERIE.git
cd DETECTION-ANOMALIE-BATTERIE
git checkout kevin                      # (ou MOHAMED, IVANJR, STEVY)

git add .
git commit -m "description du changement"
git push

git pull origin main                    # récupérer le travail déjà fusionné des autres
```

Quand une partie est terminée : **Pull Request** de sa branche vers `main` sur GitHub, relue par un autre membre avant le merge.
