# Répartition des tâches

Projet PIC — Détection d'anomalies des batteries de véhicules électriques.
Chaque membre retrouve ici **ce qu'il doit faire**, **sur quelle branche** et **ce qu'il doit livrer aux autres**.

| Membre | Branche | Rôle | Étapes du rapport |
|---|---|---|---|
| Kabre Yann Kevin | `kevin` | Données | 3, 5 |
| Traore Mohamed | `MOHAMED` | Analyse, seuils, vote, dashboard | 4, 8 |
| Mobio Ivan Junior | `IVANJR` | ML supervisé (modèles 1 et 2) | 6 |
| Gatsoundou Junior Stevy | `STEVY` | Isolation Forest et MLOps | 6, 7, 9 |

---

## Démarrage (tout le monde)

```bash
git lfs install                       # une seule fois, pour télécharger les CSV
git clone https://github.com/AMij0101982872/DETECTION-ANOMALIE-BATTERIE.git
cd DETECTION-ANOMALIE-BATTERIE
git checkout <ma-branche>             # kevin, MOHAMED, IVANJR ou STEVY
pip install -r requirements.txt
python -m src.data_loading            # crée data/processed/dataset_batterie.parquet
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

---

## 🟦 Kevin — Données (branche `kevin`)

> **Priorité du projet : tous les modèles attendent ton étape 5.**

### Étape 3 — Valider le nettoyage (`src/cleaning.py`)
- [ ] Lancer `python -m src.data_loading` puis `python -m src.cleaning`
- [ ] Compter les lignes retirées par type d'erreur (doublons, 0 V, −400 A) et compléter le Table 4 du rapport
- [ ] Ajouter un test dans `tests/` pour le cas −400 A

### Étape 5 — Sessions et features (`src/sessions.py`)
- [ ] `ajouter_session_id(df)` : nouvelle session si plus de 10 min entre deux mesures **ou** si le véhicule change (~30 000 sessions attendues)
- [ ] Écarter les sessions trop courtes (ex. moins de 10 mesures) et documenter ce choix
- [ ] `features_par_session(df)` → `data/processed/sessions_features.parquet`, **une ligne par session**, avec exactement ces colonnes :

| Groupe | Colonnes |
|---|---|
| Identité | `vehicle_id`, `session_id`, `debut`, `fin`, `mois`, `nb_mesures` |
| Session complète | `duree_min`, `soc_debut`, `soc_fin`, `courant_moyen`, `courant_max` (valeur absolue), `temp_debut`, `temp_max`, `vitesse_temp` (°C/min), `ecart_v_moyen`, `ecart_v_max` |
| 15 premières minutes (modèle 2) | `temp_15min`, `vitesse_temp_15min`, `courant_moyen_15min`, `ecart_v_max_15min`, `soc_15min`, `session_longue` (vrai si la session dure plus de 15 min) |

- [ ] Livrer **d'abord une version sur 2 véhicules** pour débloquer les autres, puis la version complète

### Ensuite
- [ ] Aider Ivan Jr sur le tuning des modèles et l'analyse des erreurs

**Livrable :** `sessions_features.parquet` + la partie « Nettoyage et sessions » du rapport.

---

## 🟩 Mohamed — Analyse, seuils, vote, dashboard (branche `MOHAMED`)

### Étape 4 — Analyse exploratoire (`notebooks/02_exploration.ipynb`)
- [ ] Distribution de `temp_max` et de `ecart_tension` : histogrammes, percentiles 99 % et 99,9 % (justifie les seuils 45 °C, 48 °C, 0,1 V)
- [ ] Température selon le mois (effet été/hiver) et selon le courant
- [ ] Comparaison des 20 véhicules (boxplots)
- [ ] Graphiques enregistrés dans `reports/figures/` pour le rapport

### Seuils et vote (`src/models/seuils.py`, `src/models/fusion.py`)
- [ ] Coder `vote_sessions(sessions)` : 1 si `temp_max` > 45 ou `ecart_v_max` > 0,1, sinon 0
- [ ] Produire les **étiquettes** pour Ivan Jr : colonnes `label_surchauffe` et `label_cellule`
- [ ] Trancher avec l'équipe : un vote global, ou un vote par anomalie (recommandé) → adapter `fusion.py`

### Étape 8 — Dashboard (`dashboard/app.py`, lancer avec `streamlit run dashboard/app.py`)
- [ ] Page « Flotte » : classement des 20 véhicules par score 0-100
- [ ] Page « Véhicule » : historique des sessions avec le niveau 🟢 / 🟠 / 🔴, filtres par période
- [ ] Détail d'une session : quelles approches ont voté anomalie, et pourquoi
- [ ] Commencer avec des **votes fictifs**, brancher les vrais résultats ensuite

**Livrables :** graphiques, étiquettes pour le modèle 2, fusion par vote, dashboard.

---

## 🟨 Ivan Jr — ML supervisé (branche `IVANJR`)

### Pour tous les modèles
- [ ] Séparation train/test **par véhicule** (`config.TEST_VEHICLES`)
- [ ] Chaque entraînement enregistré dans **MLflow** (avec la fonction `log_run` de Stevy)
- [ ] En attendant les sessions de Kevin : travailler sur un faux tableau avec les mêmes colonnes

### Modèle 1 — Régression (`src/models/regression.py`) — 6 entraînements
- [ ] Entrées : `courant_moyen`, `soc_debut`, `temp_debut`, `mois`, `duree_min`
- [ ] Cible A : `temp_max` — Cible B : `ecart_v_max`
- [ ] Algorithmes : régression linéaire (référence), Random Forest, XGBoost — pour chaque cible
- [ ] Métriques : MAE, RMSE, R² dans un tableau comparatif
- [ ] `vote_sessions` : 1 si (réel − prévu) dépasse un seuil (ex. 99e percentile des résidus) — justifier ce choix

### Modèle 2 — Alerte précoce (`src/models/alerte_precoce.py`) — 4 entraînements
- [ ] Entrées : **uniquement** les colonnes `*_15min`, sur les sessions longues
- [ ] Cibles : `label_surchauffe` et `label_cellule` (fournies par Mohamed)
- [ ] Algorithmes : Random Forest, XGBoost en classification, `class_weight="balanced"` (peu d'anomalies)
- [ ] Métriques : précision, rappel, F1, matrice de confusion — **le rappel est prioritaire**
- [ ] Ajuster le seuil de probabilité pour favoriser le rappel

**Livrable :** tableau des 10 modèles, meilleur modèle par cible, importance des variables (graphique).

---

## 🟥 Stevy — Isolation Forest et MLOps (branche `STEVY`)

> **À commencer tout de suite :** MLflow, Docker et la CI ne dépendent pas des sessions.

### Modèle 3 — Isolation Forest (`src/models/isolation_forest.py`)
- [ ] Entrées : toutes les features de session, normalisées (`StandardScaler`)
- [ ] Sortie : score de bizarrerie par session + `vote_sessions` (1 si atypique)
- [ ] Choisir et justifier `contamination` (ex. 1 % ou 2 %)
- [ ] **Évaluation :** injecter de fausses anomalies dans des sessions normales (`temp_max` + 8 °C, `ecart_v_max` + 60 mV) et mesurer le pourcentage retrouvé

### Étape 7 — MLflow et Docker
- [ ] Mettre en place MLflow et écrire une fonction `log_run(...)` utilisée par tout le monde
- [ ] Enregistrer les 5 modèles finaux dans le **Model Registry**
- [ ] `Dockerfile` : installe `requirements.txt` et lance le dashboard

### Étape 9 — CI/CD et monitoring
- [ ] `.github/workflows/ci.yml` : à chaque push → installation → `pytest` → build Docker
- [ ] Bonus **Evidently** : rapport de dérive entre 2019-2020 et 2021 sur `temp_max` et `ecart_v_max`

**Livrables :** modèle 3 évalué, MLflow, Docker, CI GitHub Actions.

---

## Qui attend quoi

```
Kevin ───── sessions_features.parquet ─────► Ivan Jr, Stevy, Mohamed
Mohamed ─── étiquettes (seuils) ───────────► Ivan Jr (modèle 2)
Stevy ───── log_run MLflow ────────────────► Ivan Jr
Ivan Jr + Stevy + Mohamed ── votes ────────► Mohamed (fusion + dashboard)
```

## Ordre de travail

| Phase | Kevin | Mohamed | Ivan Jr | Stevy |
|---|---|---|---|---|
| **1. En parallèle** | Sessions + features (priorité) | Analyse exploratoire | Code des modèles sur faux tableau | MLflow, Docker, CI |
| **2. Sessions livrées** | Aide Ivan Jr | Vote des seuils, étiquettes, début dashboard | Entraîne les 10 modèles | Isolation Forest + anomalies simulées |
| **3. Intégration** | Rapport | Fusion des 4 votes, dashboard final | Rapport + choix des modèles | Registry MLflow, Docker final, Evidently |

Point d'équipe **une fois par semaine** : où en est chacun, qui est bloqué.
