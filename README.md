# Détection d'anomalies des batteries de véhicules électriques

Projet 3pH · PIC — Filière Big Data & IA, HESTIM (2026-2027)

**Problématique :** comment détecter, expliquer et anticiper automatiquement la **surchauffe** et la **cellule défaillante** d'une batterie de véhicule électrique, à partir de millions de mesures réelles et sans étiquettes de panne ?

👉 **Qui fait quoi, et par où commencer : [TACHE.md](TACHE.md)**

## Les deux datasets (utilisés côte à côte, jamais fusionnés)

| | **BAIC** — dataset principal | **Zhang et al.** — validation |
|---|---|---|
| Rôle | Tout le pipeline : 9 étapes, 4 approches, vote, dashboard | Prouver que l'indicateur « cellule » détecte de **vraies pannes** |
| Source | Deng et al., *Applied Energy*, 2023 | Zhang et al., *Nature Communications*, 2023 (CC BY 4.0) |
| Contenu | 16,1 M de mesures, 20 véhicules BAIC EU500, 2019 → 2021 | 700 582 extraits, 347 véhicules (3 marques), **55 en panne** |
| Étiquettes | Aucune (fabriquées par les seuils) | Vraies pannes, par véhicule |
| Fichier | `data/raw/#1.csv … #20.csv` (Git LFS) | `data/external/zhang_extraits.parquet` (10 Mo, dans le dépôt) |
| Anomalies | Surchauffe **et** cellule | **Cellule uniquement** (la température n'y sépare pas les pannes) |

**Pourquoi on ne les fusionne pas :** unités différentes (session complète vs extrait de 21 min), marques et capteurs différents (écart normal ≈ 41 mV chez BAIC, ≈ 20 mV chez Zhang). Un modèle entraîné sur le mélange apprendrait à reconnaître *le dataset*, pas *la panne*. Le lien entre les deux est **l'indicateur** (écart de tension moyen entre cellules) : validé sur Zhang, appliqué aux BAIC.

## Les 9 étapes du pipeline (sur BAIC)

| # | Étape | Statut | Code | Qui |
|---|---|---|---|---|
| 1 | Collecte des données | ✅ Réalisé | `data/raw/` | — |
| 2 | Fusion et stockage | ✅ Réalisé | `src/data_loading.py` | Kevin |
| 3 | Contrôle qualité et nettoyage | ✅ Réalisé (branche `kevin`, à fusionner) | `src/quality.py`, `src/cleaning.py` | Kevin |
| 4 | Analyse exploratoire | 🟠 En cours | `notebooks/` | Mohamed |
| 5 | Sessions de charge et features | ✅ Réalisé (branche `kevin`, à fusionner) | `src/sessions.py` | Kevin |
| 6 | Modèles de machine learning | ⬜ À faire | `src/models/` | Ivan Jr, Stevy, Mohamed |
| 7 | Sauvegarde et intégration (MLflow, Docker) | ⬜ À faire | `Dockerfile`, MLflow | Stevy |
| 8 | Dashboard de supervision | ⬜ À faire | `dashboard/app.py` | Mohamed |
| 9 | Supervision et amélioration continue | ⬜ À faire | `tests/`, `.github/workflows/`, Evidently | Stevy |

## Les 4 approches de détection + fusion par vote

| Approche | Type | Surchauffe | Cellule | Fichier |
|---|---|---|---|---|
| Seuils — « ça dépasse maintenant » | Règles | 45 °C / 48 °C | 60 mV / 100 mV ¹ | `src/models/seuils.py` |
| Modèle 1 : Régression — « elle chauffe plus que prévu » | Supervisé | ✅ | ✅ | `src/models/regression.py` |
| Modèle 2 : Alerte précoce — « ça va arriver » | Supervisé | ✅ | ❌ ¹ | `src/models/alerte_precoce.py` |
| Modèle 3 : Atypique — « cette charge est bizarre » | Non supervisé | ✅ | ✅ | `src/models/isolation_forest.py` |
| **Fusion par vote** | — | 0-1 normal · 2 à surveiller · 3-4 critique | 3 votes ¹ | `src/models/fusion.py` |
| Validation sur vraies pannes (Zhang) | Supervisé | — | ✅ ¹ | `src/models/cellule_zhang.py` |

¹ Selon la **[décision sur le seuil de la cellule](DECISION_SEUIL_CELLULE.md)** (options C/D) : à 0,1 V aucune session BAIC n'est anormale, et 90 % des cas à 60 mV viennent du véhicule 9 → pas de modèle 2 pour la cellule. **Chacun doit voter dans ce document.**

**Résultat clé (Zhang) :** l'écart de tension moyen sépare les véhicules en panne des normaux — AUROC **0,77** (marque 1), **0,90** (marque 2), 0,57 (marque 3). Règle « écart moyen > 40 mV » : 53 % des pannes détectées pour 3 % de fausses alertes. → Utiliser l'écart **moyen**, pas le maximum.

## Démarrer

```bash
git lfs install                         # une seule fois, avant le clone (CSV de data/raw/)
git clone https://github.com/AMij0101982872/DETECTION-ANOMALIE-BATTERIE.git
cd DETECTION-ANOMALIE-BATTERIE
git checkout <ma-branche>               # kevin, MOHAMED, IVANJR ou STEVY
git pull origin main                    # récupérer le dernier état commun
pip install -r requirements.txt

python main.py                          # fusion → nettoyage → sessions (crée data/processed/)
python -m pytest                        # tests
```

Autres commandes :

```bash
python -m src.quality                   # rapport qualité par véhicule
python -m src.models.seuils             # résumé des anomalies par seuils
streamlit run dashboard/app.py          # dashboard
```

Lire les données dans Python :

```python
import pandas as pd
sessions = pd.read_parquet("data/processed/sessions_features.parquet")    # BAIC, 1 ligne = 1 session
zhang = pd.read_parquet("data/external/zhang_extraits.parquet")           # Zhang, 1 ligne = 1 extrait
```

## Équipe

| Membre | Branche | Rôle | Étapes |
|---|---|---|---|
| **1 — Kabre Yann Kevin** | `kevin` | Données | 2, 3, 5 |
| **2 — Traore Mohamed** | `MOHAMED` | Analyse, seuils, vote, dashboard | 4, 8 |
| **3 — Mobio Ivan Junior** | `IVANJR` | ML supervisé (modèles 1, 2, validation Zhang) | 6 |
| **4 — Gatsoundou Junior Stevy** | `STEVY` | Isolation Forest et MLOps | 6, 7, 9 |

## Règles d'évaluation (à respecter par tous)

- **Séparation par véhicule** : entraînement sur 16 véhicules, test sur 4 jamais vus (`config.TEST_VEHICLES` = 17, 18, 19, 20). Sur Zhang : `split` train/test fourni.
- **Anomalie confirmée** : au moins 3 mesures consécutives.
- **Modèle 2** : ne voit que les 15 premières minutes de la session (sinon raisonnement circulaire).
- **Erreurs capteur** : retirées au nettoyage, jamais comptées comme anomalies.
- **Déséquilibre (~1 % d'anomalies)** : pas d'accuracy ; rappel, F1, PR-AUC, `class_weight="balanced"`.

## Architecture

```
Dataset_automobile/
├── data/raw/              # BAIC : #1.csv … #20.csv (Git LFS)
├── data/processed/        # parquet fusionné, nettoyé, sessions (régénérés par main.py, non versionnés)
├── data/external/         # Zhang : zhang_extraits.parquet + LISEZMOI.md (source, licence)
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
├── DECISION_SEUIL_CELLULE.md
├── TACHE.md
├── reports/figures/
└── requirements.txt
```

## Travail en équipe (Git)

Chaque membre travaille sur **sa** branche, jamais directement sur `main` :

```bash
git add .
git commit -m "description du changement"
git push
git pull origin main                    # régulièrement : récupérer le travail fusionné des autres
```

Quand une partie est terminée : **Pull Request** de sa branche vers `main` sur GitHub, relue par un autre membre, `python -m pytest` doit passer.
