# Répartition des tâches

Projet PIC — Détection d'anomalies des batteries de véhicules électriques.
Chaque membre retrouve ici **ce qu'il doit faire**, **sur quelle branche** et **ce qu'il doit livrer aux autres**.

| Membre | Branche | Rôle | Étapes du rapport |
|---|---|---|---|
| Kabre Yann Kevin | `kevin` | Données | 2, 3, 5 |
| Traore Mohamed | `MOHAMED` | Analyse, seuils, vote, dashboard | 4, 8 |
| Mobio Ivan Junior | `IVANJR` | ML supervisé (modèles 1, 2, validation Zhang) | 6 |
| Gatsoundou Junior Stevy | `STEVY` | Isolation Forest et MLOps | 6, 7, 9 |

---

## 🚀 Par où commencer (maintenant)

| Ordre | Qui | Action | Pourquoi |
|---|---|---|---|
| ✅ | Kevin | Pull Request `kevin` → `main` **fusionnée** : nettoyage + sessions sur `main` | — |
| ✅ | Ivan Jr | **Modèle 1 (régression) sur `main`** : votes dans `votes_regression.parquet` | — |
| **1** | **Tous** | `git pull origin main` puis `python main.py` et `python -m src.models.regression` | Crée les sessions et les votes du modèle 1 chez chacun |
| **2** | **Stevy** | Ouvrir la **Pull Request `STEVY` → `main`** (`log_run`, Isolation Forest, Docker, CI) | `log_run` et le vote du modèle 3 sont attendus par Ivan Jr et Mohamed |
| **3** | **Chacun** | Sa tâche marquée ⭐ ci-dessous | — |

> ✅ **Décision prise : option D** ([DECISION_SEUIL_CELLULE.md](DECISION_SEUIL_CELLULE.md)) : seuils cellule à deux niveaux (alerte 60 mV / critique 100 mV), **pas de modèle 2 pour la cellule**, validation sur les vraies pannes de Zhang. Les tâches marquées **🅳** viennent de cette décision.

---

## Démarrage (tout le monde)

```bash
git lfs install                       # une seule fois, pour télécharger les CSV
git clone https://github.com/AMij0101982872/DETECTION-ANOMALIE-BATTERIE.git
cd DETECTION-ANOMALIE-BATTERIE
git checkout <ma-branche>             # kevin, MOHAMED, IVANJR ou STEVY
git pull origin main                  # dernier état commun (docs, dataset Zhang, sessions après merge)
pip install -r requirements.txt
python main.py                        # fusion → nettoyage → sessions (data/processed/)
```

Travail au quotidien :

```bash
git add .
git commit -m "ce que j'ai fait"
git push
git pull origin main                  # récupérer ce que les autres ont déjà fusionné
```

## Règles communes

- Chacun travaille sur **sa branche** et ne modifie pas les fichiers des autres.
- Seuils, chemins et paramètres sont dans `src/config.py` : **on ne les change pas sans prévenir le groupe.**
- Le **format des fichiers échangés** (voir chaque section) est un contrat : si on le change, on prévient.
- Avant chaque Pull Request vers `main` : `python -m pytest` doit passer, et un autre membre relit.
- Train/test **par véhicule** pour tous les modèles : `config.TEST_VEHICLES` = 17, 18, 19, 20 en test.
- Classes déséquilibrées (~1 %) : **pas d'accuracy** ; rappel, F1, PR-AUC, `class_weight="balanced"`.

## Les deux datasets

| | BAIC (`data/raw/`) | Zhang et al. (`data/external/zhang_extraits.parquet`) |
|---|---|---|
| Rôle | **Tout le projet** (surchauffe + cellule) | **Validation** de l'indicateur cellule sur de vraies pannes |
| Une ligne | une session de charge (après étape 5) | un extrait de 21 min |
| Étiquettes | aucune (fabriquées par les seuils) | `label` = 1 si véhicule en panne réelle |

- On **ne fusionne pas** les deux datasets (unités, marques et capteurs différents).
- Zhang : **cellule uniquement** (la température n'y sépare pas les pannes). Utiliser l'écart **moyen**, pas le maximum.
- Licence CC BY 4.0 : **citer Zhang et al. 2023** (voir `data/external/LISEZMOI.md`). Les 7 Go de `.pkl` bruts ne vont **jamais** sur GitHub, et ne s'ouvrent **jamais** avec `pickle.load` / `torch.load`.

---

## 🟦 Kevin — Données (branche `kevin`)

### Étapes 3 et 5 — ✅ faites sur ta branche
- [x] Nettoyage : comptage par type d'erreur + tests (`src/cleaning.py`, `tests/test_cleaning.py`)
- [x] Sessions et features : 22 colonnes, `sessions_features.parquet` (`src/sessions.py`, `tests/test_sessions.py`)
- [x] Pull Request `kevin` → `main` fusionnée
- [ ] ⭐ `src/external_zhang.py` (voir plus bas)
- [ ] Vérifier les 16 sessions de plus de 12 h (jusqu'à 65 h) : vraie charge ou deux sessions collées ?
- [ ] Compléter le Table 4 du rapport (lignes retirées par type d'erreur)

**Contrat de `sessions_features.parquet`** (une ligne par session) :

| Groupe | Colonnes |
|---|---|
| Identité | `vehicle_id`, `session_id`, `debut`, `fin`, `mois`, `nb_mesures` |
| Session complète | `duree_min`, `soc_debut`, `soc_fin`, `courant_moyen`, `courant_max` (valeur absolue), `temp_debut`, `temp_max`, `vitesse_temp` (°C/min), `ecart_v_moyen`, `ecart_v_max` |
| 15 premières minutes (modèle 2) | `temp_15min`, `vitesse_temp_15min`, `courant_moyen_15min`, `ecart_v_max_15min`, `soc_15min`, `session_longue` |

### 🅳 Dataset Zhang (`src/external_zhang.py`)
- [x] `data/external/zhang_extraits.parquet` produit (700 582 extraits, 347 véhicules, 55 en panne) et dans le dépôt
- [ ] Intégrer le script d'export dans `src/external_zhang.py` (lecteur **sécurisé** : `numpy` et `OrderedDict` uniquement), pour que le fichier soit reproductible
- [ ] Un test : le lecteur refuse un pickle qui appelle une fonction interdite

### Ensuite
- [ ] Aider Ivan Jr sur le tuning des modèles et l'analyse des erreurs
- [ ] Rédiger la partie « Données, nettoyage et sessions » du rapport

---

## 🟩 Mohamed — Analyse, seuils, vote, dashboard (branche `MOHAMED`)

### Étape 4 — Analyse exploratoire (`notebooks/02_exploration.ipynb`)
- [ ] ⭐ Distribution de `temp_max`, `ecart_v_moyen` et `ecart_v_max` par session : histogrammes, percentiles 99 % et 99,9 % (justifie 45/48 °C et 60/100 mV)
- [ ] Température selon le mois (effet été/hiver) et selon le courant
- [ ] Comparaison des 20 véhicules (boxplots) — montrer que le **véhicule 9** se détache sur l'écart de tension
- [ ] Graphiques enregistrés dans `reports/figures/` pour le rapport

### Seuils et vote (`src/models/seuils.py`, `src/models/fusion.py`)
- [ ] Coder `vote_sessions(sessions)` **par anomalie** :
  - surchauffe : alerte si `temp_max` > 45, critique si ≥ 48
  - 🅳 cellule : alerte si `ecart_v_max` > 60 mV, critique si > 100 mV (tester aussi `ecart_v_moyen`)
- [ ] Produire l'**étiquette** pour Ivan Jr : colonne `label_surchauffe` (🅳 plus besoin de `label_cellule`)
- [ ] ⭐ 🅳 Adapter `fusion.py` : vote par anomalie, **4 votes pour la surchauffe, 3 pour la cellule**, en lisant les fichiers de votes (voir « Contrat des votes » plus bas). Le modèle 1 est **déjà disponible**.
- [ ] `config.py` : ajouter `ECART_TENSION_ALERTE = 0.06` (prévenir le groupe)

### Étape 8 — Dashboard (`dashboard/app.py`, lancer avec `streamlit run dashboard/app.py`)
- [ ] Commencer avec des **votes fictifs**, brancher les vrais résultats ensuite
- [ ] Page « Flotte » : classement des 20 véhicules par score 0-100
- [ ] Page « Véhicule » : historique des sessions avec le niveau 🟢 / 🟠 / 🔴, filtres par période
- [ ] Détail d'une session : quelles approches ont voté anomalie, et pourquoi
- [ ] Page « Véhicule 9 » : évolution de son écart de tension dans le temps
- [ ] 🅳 Page « Validation sur vraies pannes » : écart moyen en panne vs normaux (Zhang), AUROC par marque, résultats d'Ivan Jr

**Livrables :** graphiques, étiquette `label_surchauffe`, fusion par vote, dashboard.

---

## 🟨 Ivan Jr — ML supervisé (branche `IVANJR`)

### Pour tous les modèles
- [x] Séparation train/test **par véhicule** (`config.TEST_VEHICLES` ; sur Zhang : colonne `split`) — appliquée au modèle 1
- [ ] Chaque entraînement enregistré dans **MLflow** (avec la fonction `log_run` de Stevy)

### ✅ Modèle 1 — Régression (`src/models/regression.py`, `notebooks/03_modele1_regression.ipynb`) — 6 entraînements
- [x] Entrées : `courant_moyen`, `soc_debut`, `temp_debut`, `mois`, `duree_min`
- [x] Cible A : `temp_max` — Cible B : `ecart_v_max`
- [x] Algorithmes : régression linéaire (référence), Random Forest, XGBoost — pour chaque cible ; choix du meilleur par validation croisée par véhicule → **XGBoost** pour les deux
- [x] Métriques : MAE, RMSE, R² → `temp_max` : MAE 0,52 °C, R² 0,98 · `ecart_v_max` : MAE 4,9 mV, R² 0,30
- [x] `vote_sessions` : 1 si (réel − prévu) > 99e percentile des résidus hors-échantillon (1,63 °C / 16,1 mV) → `data/processed/votes_regression.parquet`
- [x] 3 tests (`tests/test_regression.py`), 9 graphiques (`reports/figures/regression_*.png`)
- [ ] Enregistrer les runs dans MLflow quand `log_run` (Stevy) sera sur `main` (le code l'appelle déjà automatiquement)

### ✅ Modèle 2 — Alerte précoce (`src/models/alerte_precoce.py`, `notebooks/05_modele2_alerte_precoce.ipynb`) — surchauffe uniquement 🅳
- [x] Population : sessions > 15 min pas encore > 45 °C à 15 min (30 216 sessions, 295 surchauffes, 1 %) ; déjà > 45 °C à 15 min → vote 1 ; < 15 min → vote 0
- [x] Entrées : uniquement ce qui est connu à 15 min (`*_15min` + `temp_debut`, `soc_debut`, `mois`)
- [x] Règle de référence (`temp_15min` ≥ 42 °C), régression logistique, Random Forest, XGBoost ; classes pondérées ; hyperparamètres par GridSearch (GroupKFold) → **régression logistique** retenue (meilleure PR-AUC en validation croisée : 0,83, la plus simple)
- [x] Seuil de probabilité choisi par F2 (rappel prioritaire) en validation croisée par véhicule
- [x] Test : **46 surchauffes sur 53 détectées (rappel 87 %)**, précision 61 %, 29 fausses alertes (règle : rappel 72 %, précision 22 %, 138 fausses alertes)
- [x] Avance : l'alerte tombe **≈ 32 min (médiane) avant** le passage à 45 °C
- [x] Notebook académique : EDA, fuite de données, déséquilibre, baselines, validation croisée, GridSearch, analyse des erreurs, coefficients et importance par permutation
- [x] Votes dans `data/processed/votes_alerte_precoce.parquet`, 3 tests (`tests/test_alerte_precoce.py`)

### ⭐ 🅳 Validation cellule sur vraies pannes (`src/models/cellule_zhang.py`) — après le modèle 2
- [ ] Charger `data/external/zhang_extraits.parquet`, agréger **par véhicule** : écart moyen, 99e percentile, max, nombre d'extraits
- [ ] Évaluer la règle simple « écart moyen > seuil » : rappel, précision, AUROC, **par marque**
- [ ] Classifieur Random Forest / XGBoost sur les vraies étiquettes, entraîné sur `split == "train"`, testé sur `split == "test"` (marques 1 et 2 ; marque 3 n'a qu'un dossier `data`)
- [ ] Signaler les limites : étiquettes par véhicule, ≈ 6× moins d'extraits pour les véhicules en panne, marque 3 non séparée (AUROC 0,57), seuils non transférables aux BAIC

**Livrables :** ✅ modèle 1 · ✅ modèle 2 · tableau des 8 modèles BAIC, meilleur modèle par cible, importance des variables (graphique) + résultats de validation sur vraies pannes (Zhang).

---

## 🟥 Stevy — Isolation Forest et MLOps (branche `STEVY`)

> **À commencer tout de suite :** MLflow, Docker et la CI ne dépendent pas des sessions.

### Étape 7 — MLflow et Docker
- [x] MLflow + `log_run(...)` dans `src/mlops.py` (branche `STEVY`)
- [ ] ⭐ **Ouvrir la Pull Request `STEVY` → `main`** : Ivan Jr et Mohamed en ont besoin
- [x] `Dockerfile` : installe `requirements.txt` et lance le dashboard (branche `STEVY`)
- [x] Modèle 3 inscrit dans le **Model Registry** (`isolation_forest`, rechargé à l'identique)
- [ ] Inscrire aussi les modèles finaux d'Ivan Jr (après son choix des modèles)

### Étape 9 — CI/CD et monitoring
- [x] `.github/workflows/ci.yml` : à chaque push → installation → `pytest` → build Docker (branche `STEVY`)
- [ ] Bonus **Evidently** : rapport de dérive entre 2019-2020 et 2021 sur `temp_max` et `ecart_v_max` (exemples tout trouvés : vieillissement de la flotte +9 mV, véhicule 9)

### Modèle 3 — Isolation Forest unilatéral (`src/models/isolation_forest.py`) — ✅ terminé
Démarche, hypothèses et résultats : [notebooks/04_modele3_isolation_forest.ipynb](notebooks/04_modele3_isolation_forest.ipynb)
- [x] Entrées : `temp_max`, `vitesse_temp`, `ecart_v_max` — **pas** toutes les features : 10 colonnes diluent le signal (rappel 7 % contre 88 %), `ecart_v_moyen` est redondant
- [x] Écrêtage unilatéral à la médiane : seules les valeurs **hautes** votent (38 % des votes venaient de valeurs basses sans danger)
- [x] `contamination` = 2 % (justifiée : +30 points de rappel de 1 à 2 %, +1,7 de 2 à 3 %)
- [x] Sessions < 5 min non évaluables (artefacts) : vote 0 et `if_evaluable` = faux
- [x] Votes dans `data/processed/votes_isolation_forest.parquet` (voir « Contrat des votes »)
- [x] **Évaluation** par anomalies injectées : surchauffe +6 °C → 75 % (seuils 23 %), cellule +30 mV → 93 %, combinée +4 °C et +20 mV → 83 % (seuils 63 %), fausses alarmes 1,5 %
- [x] **Véhicule 9** : 1ᵉʳ / 20 (14 % de ses sessions), suit sa dégradation de 2021
- [ ] 🅳 Suivre dans MLflow l'expérience « cellule_zhang » d'Ivan Jr
- [ ] Signaler à Kevin : `main.py` n'applique pas `filtrer_sessions_courtes` (180 sessions de moins de 10 mesures)

**Livrables :** modèle 3 évalué, MLflow, Docker, CI GitHub Actions.

---

## Contrat des votes (pour la fusion de Mohamed)

Chaque approche écrit **un fichier de votes**, une ligne par session, avec `vehicle_id`, `session_id` et ses colonnes de vote (0/1). Mohamed les joint sur `session_id`.

| Approche | Fichier | Colonnes de vote | Anomalies | Statut |
|---|---|---|---|---|
| Seuils (Mohamed) | `data/processed/votes_seuils.parquet` | `vote_seuils_surchauffe`, `vote_seuils_cellule` | les deux | à faire |
| Modèle 1 (Ivan Jr) | `data/processed/votes_regression.parquet` | `vote_regression_surchauffe`, `vote_regression_cellule` | les deux | ✅ sur `main` |
| Modèle 2 (Ivan Jr) | `data/processed/votes_alerte_precoce.parquet` | `vote_alerte_precoce_surchauffe` (+ `proba_alerte_precoce`, `ap_evaluable`, `raison_alerte_precoce`) | surchauffe | ✅ branche `IVANJR` |
| Modèle 3 (Stevy) | `data/processed/votes_isolation_forest.parquet` | `vote_isolation_forest` (+ `if_evaluable`, `score_isolation_forest`, `raison_isolation_forest` pour le dashboard) | les deux (un seul vote) | ✅ branche `STEVY` (PR) |

Fusion (option D) : **surchauffe = 4 votes** (seuils, modèle 1, modèle 2, modèle 3) · **cellule = 3 votes** (seuils, modèle 1, modèle 3).

## Qui attend quoi

```
Kevin ───── sessions_features.parquet ✅ ───────────────► Ivan Jr, Stevy, Mohamed
Stevy ───── PR STEVY → main (log_run, votes modèle 3) ──► Ivan Jr, Mohamed
Mohamed ─── label_surchauffe ────────────────────────────► Ivan Jr (modèle 2)
Ivan Jr + Stevy + Mohamed ── votes ──────────────────────► Mohamed (fusion + dashboard)
Ivan Jr ─── résultats cellule_zhang ─────────────────────► Mohamed (page validation)
```

## Ordre de travail

| Phase | Kevin | Mohamed | Ivan Jr | Stevy |
|---|---|---|---|---|
| **1. Maintenant** | `external_zhang.py`, sessions > 12 h | Analyse exploratoire, vote des seuils | ✅ Modèle 1 · Modèle 2 | PR `STEVY` → `main`, votes du modèle 3 |
| **2. Votes disponibles** | Aide Ivan Jr, rapport | Fusion des votes, début dashboard | Validation cellule sur Zhang | Registry MLflow, véhicule 9 |
| **3. Intégration** | Rapport (données) | Fusion des votes, dashboard final | Rapport + choix des modèles | Registry MLflow, Docker final, Evidently |

Point d'équipe **une fois par semaine** : où en est chacun, qui est bloqué.
