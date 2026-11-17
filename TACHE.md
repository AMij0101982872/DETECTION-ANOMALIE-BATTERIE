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
| **1** | **Tous** | Lire [DECISION_SEUIL_CELLULE.md](DECISION_SEUIL_CELLULE.md) et **voter** (A, B, C ou D) | Le seuil cellule change le travail de Mohamed et d'Ivan Jr |
| **2** | **Kevin** | Ouvrir la **Pull Request `kevin` → `main`** (nettoyage + sessions, 11 tests OK) | **Débloque tout le monde** : les sessions ne sont pas encore sur `main` |
| **3** | **Tous** | Après le merge : `git pull origin main` puis `python main.py` | Crée `data/processed/sessions_features.parquet` chez chacun |
| **4** | **Chacun** | Sa première tâche ci-dessous (⭐) | — |

> En attendant le vote, on travaille avec l'**option D** (proposée) : seuils cellule à deux niveaux (alerte 60 mV / critique 100 mV), **pas de modèle 2 pour la cellule**, validation sur les vraies pannes de Zhang. Les tâches marquées **🅳** en dépendent.

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
- [ ] ⭐ **Ouvrir la Pull Request `kevin` → `main`** (priorité n° 1 du projet)
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
- [ ] 🅳 Adapter `fusion.py` : vote par anomalie, **4 votes pour la surchauffe, 3 pour la cellule**
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
- [ ] Séparation train/test **par véhicule** (`config.TEST_VEHICLES` ; sur Zhang : colonne `split`)
- [ ] Chaque entraînement enregistré dans **MLflow** (avec la fonction `log_run` de Stevy)

### ⭐ 🅳 Validation cellule sur vraies pannes (`src/models/cellule_zhang.py`) — à faire en premier (ne dépend de personne)
- [ ] Charger `data/external/zhang_extraits.parquet`, agréger **par véhicule** : écart moyen, 99e percentile, max, nombre d'extraits
- [ ] Évaluer la règle simple « écart moyen > seuil » : rappel, précision, AUROC, **par marque**
- [ ] Classifieur Random Forest / XGBoost sur les vraies étiquettes, entraîné sur `split == "train"`, testé sur `split == "test"` (marques 1 et 2 ; marque 3 n'a qu'un dossier `data`)
- [ ] Signaler les limites : étiquettes par véhicule, ≈ 6× moins d'extraits pour les véhicules en panne, marque 3 non séparée (AUROC 0,57), seuils non transférables aux BAIC

### Modèle 1 — Régression (`src/models/regression.py`) — 6 entraînements (après le merge des sessions)
- [ ] Entrées : `courant_moyen`, `soc_debut`, `temp_debut`, `mois`, `duree_min`
- [ ] Cible A : `temp_max` — Cible B : `ecart_v_max`
- [ ] Algorithmes : régression linéaire (référence), Random Forest, XGBoost — pour chaque cible
- [ ] Métriques : MAE, RMSE, R² dans un tableau comparatif
- [ ] `vote_sessions` : 1 si (réel − prévu) dépasse un seuil (ex. 99e percentile des résidus) — justifier ce choix

### Modèle 2 — Alerte précoce (`src/models/alerte_precoce.py`) — surchauffe uniquement 🅳
- [ ] Entrées : **uniquement** les colonnes `*_15min`, sur les sessions longues (`session_longue`)
- [ ] Cible : `label_surchauffe` (fournie par Mohamed ; en attendant, la calculer : `temp_max > 45`)
- [ ] Algorithmes : Random Forest, XGBoost en classification, `class_weight="balanced"` / `scale_pos_weight`
- [ ] Métriques : précision, rappel, F1, PR-AUC, matrice de confusion — **le rappel est prioritaire**
- [ ] Ajuster le seuil de probabilité pour favoriser le rappel

**Livrables :** tableau des 8 modèles BAIC, meilleur modèle par cible, importance des variables (graphique) + résultats de validation sur vraies pannes (Zhang).

---

## 🟥 Stevy — Isolation Forest et MLOps (branche `STEVY`)

> **À commencer tout de suite :** MLflow, Docker et la CI ne dépendent pas des sessions.

### Étape 7 — MLflow et Docker
- [ ] ⭐ Mettre en place MLflow et écrire une fonction `log_run(...)` utilisée par tout le monde (à livrer tôt : Ivan Jr en a besoin)
- [ ] `Dockerfile` : installe `requirements.txt` et lance le dashboard
- [ ] Enregistrer les modèles finaux dans le **Model Registry**

### Étape 9 — CI/CD et monitoring
- [ ] `.github/workflows/ci.yml` : à chaque push → installation → `pytest` → build Docker
- [ ] Bonus **Evidently** : rapport de dérive entre 2019-2020 et 2021 sur `temp_max` et `ecart_v_max`

### Modèle 3 — Isolation Forest (`src/models/isolation_forest.py`) — après le merge des sessions
- [ ] Entrées : toutes les features de session, normalisées (`StandardScaler`)
- [ ] Sortie : score de bizarrerie par session + `vote_sessions` (1 si atypique)
- [ ] Choisir et justifier `contamination` (ex. 1 % ou 2 %)
- [ ] **Évaluation :** injecter de fausses anomalies dans des sessions normales (`temp_max` + 8 °C, `ecart_v_max` + 60 mV) et mesurer le pourcentage retrouvé
- [ ] Vérifier qu'il repère bien le **véhicule 9**
- [ ] 🅳 Suivre dans MLflow l'expérience « cellule_zhang » d'Ivan Jr

**Livrables :** modèle 3 évalué, MLflow, Docker, CI GitHub Actions.

---

## Qui attend quoi

```
Kevin ───── PR kevin → main (sessions_features.parquet) ──► Ivan Jr, Stevy, Mohamed   ← BLOQUANT
Stevy ───── log_run MLflow ──────────────────────────────► Ivan Jr
Mohamed ─── label_surchauffe ────────────────────────────► Ivan Jr (modèle 2)
Ivan Jr + Stevy + Mohamed ── votes ──────────────────────► Mohamed (fusion + dashboard)
Ivan Jr ─── résultats cellule_zhang ─────────────────────► Mohamed (page validation)
```

## Ordre de travail

| Phase | Kevin | Mohamed | Ivan Jr | Stevy |
|---|---|---|---|---|
| **1. Maintenant** | PR `kevin` → `main`, sessions > 12 h | Analyse exploratoire, vote du seuil | Validation cellule sur Zhang | MLflow (`log_run`), Docker, CI |
| **2. Sessions sur `main`** | `external_zhang.py`, aide Ivan Jr | Vote des seuils, `label_surchauffe`, début dashboard | Modèles 1 et 2 (8 entraînements) | Isolation Forest + anomalies simulées |
| **3. Intégration** | Rapport (données) | Fusion des votes, dashboard final | Rapport + choix des modèles | Registry MLflow, Docker final, Evidently |

Point d'équipe **une fois par semaine** : où en est chacun, qui est bloqué.
