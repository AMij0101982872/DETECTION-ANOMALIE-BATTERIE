# Décision à prendre : seuil de la cellule défaillante

> **Statut : EN ATTENTE DE DÉCISION**. Chacun lit, puis vote en bas du document.
> Tant que ce n'est pas tranché, Mohamed (vote des seuils) et Ivan Jr (modèle 2) ne peuvent pas finaliser leur partie.

---

## 1. Le problème en une phrase

Avec le seuil actuel de **0,1 V**, **aucune** des 30 855 sessions n'est une « cellule défaillante ». On ne peut donc ni voter avec les seuils, ni entraîner le modèle 2 (alerte précoce) pour cette anomalie.

## 2. Les chiffres (sessions de Kevin, `sessions_features.parquet`)

**Écart de tension maximal par session (`ecart_v_max`) :**

| Médiane | 90 % | 99 % | 99,9 % | Maximum |
|---|---|---|---|---|
| 41 mV | 49 mV | 57 mV | 73 mV | **98 mV** |

→ Aucune session n'atteint 100 mV.

**Comparaison avec la surchauffe (sessions longues, plus de 15 min) :**

| | Surchauffe (> 45 °C) | Cellule (> 0,1 V) | Cellule (> 60 mV) |
|---|---|---|---|
| Sessions anormales | 301 (1 %) | **0** | 190 (0,6 %) |
| Dans le train (16 véhicules) | 246 | 0 | 182 |
| Dans le test (véhicules 17 à 20) | 55 | 0 | **8** |
| Véhicules concernés | **les 20** (8 à 25 chacun) | aucun | **surtout le 9** |

**Répartition des 190 sessions au-delà de 60 mV :**

| Véhicule 9 | Véhicule 13 | Véhicule 18 | Véhicule 6 | Les 16 autres |
|---|---|---|---|---|
| **171** | 10 | 8 | 1 | 0 |

## 3. Pourquoi c'est un problème

1. **À 0,1 V** : 0 exemple. Un modèle ne peut pas apprendre ce qu'il n'a jamais vu.
2. **À 60 mV** : 90 % des exemples viennent du **véhicule 9**. Le modèle 2 apprendrait à reconnaître *le véhicule 9*, pas les signes d'une cellule qui faiblit. C'est exactement ce que la séparation par véhicule (rapport §3.7) veut éviter. Et avec 8 cas en test, le rappel ne serait pas fiable.
3. **Le déséquilibre (1 %) n'est PAS le problème** : il est normal en détection d'anomalies et se gère (`class_weight`, rappel, PR-AUC). La surchauffe a le même déséquilibre et fonctionne.

> Le rapport (Table 1) indique « environ 16 mV sur un pack sain » : c'est faux au niveau session. La médiane est de **41 mV**. À corriger.

## 4. Les options

### Option A : garder 0,1 V et accepter le résultat
- ✅ Honnête : 0,1 V reste la limite d'ingénierie.
- ✅ Conclusion : « aucune cellule défaillante dans la flotte ».
- ❌ Pas de vote des seuils ni de modèle 2 pour la cellule. Il ne reste que la régression et l'Isolation Forest.

### Option B : remplacer 0,1 V par 60 mV
- ✅ Des exemples existent.
- ❌ Critiquable : on baisse le seuil pour *fabriquer* des anomalies. 60 mV ne veut pas dire « cellule défaillante ».
- ❌ Ne règle pas le problème du véhicule 9 pour le modèle 2.

### Option C : deux niveaux, comme pour la température (RECOMMANDÉE)

| | Surchauffe | Cellule |
|---|---|---|
| **Alerte** (relative, issue des données) | > 45 °C | **> 60 mV** (≈ 99e percentile) |
| **Critique** (absolue, issue de l'ingénierie) | ≥ 48 °C | **> 100 mV** (inchangé) |

Et, pour la cellule, **pas de modèle 2** :

| Approche | Surchauffe | Cellule |
|---|---|---|
| Seuils | ✅ | ✅ (alerte 60 mV / critique 100 mV) |
| Modèle 1 : Régression | ✅ | ✅ |
| Modèle 2 : Alerte précoce | ✅ | ❌ retiré (anomalie propre à un véhicule) |
| Modèle 3 : Isolation Forest | ✅ | ✅ |
| **Votes possibles** | 4 | 3 |

- ✅ Rien n'est caché : 0,1 V reste en place.
- ✅ Même logique que la température : cohérent devant le jury.
- ✅ **Le véhicule 9 devient un résultat** : « le système a repéré un véhicule dont une cellule se déséquilibre ». À montrer dans le dashboard, avec l'évolution de son écart de tension dans le temps.
- ⚠️ La fusion par vote doit s'adapter : 3 votes pour la cellule (2 votes ou plus = critique ?).

### Option D : option C + validation sur de vraies pannes (dataset Zhang et al.) — PROPOSÉE PAR IVAN JR

**L'idée :** notre limite principale est qu'on n'a **aucune vraie panne** dans les données BAIC. Un dataset public en contient : **Zhang et al., *Nature Communications*, 2023** (figshare `10.6084/m9.figshare.23659323`).

| | Nos BAIC | Zhang et al. |
|---|---|---|
| Véhicules | 20 (1 marque) | 347 (3 marques), dont **55 en panne confirmée** |
| Unité | Session complète | Extrait de 128 mesures (10 s, ≈ 21 min) |
| Étiquettes | Aucune (fabriquées par seuils) | **Vraies pannes**, par véhicule (0/1) |
| Colonnes | tension pack, courant, SOC, cellule max/min, temp max/min | `volt` (normalisée), `current`, `soc`, `max_single_volt`, `min_single_volt`, `max_temp`, `min_temp`, `timestamp` |

**Résultats (marque 1 complète : 198 véhicules, 30 en panne, 476 739 extraits) :**

| Indicateur (par véhicule) | Normaux | En panne | AUROC |
|---|---|---|---|
| Écart de tension moyen | 20 mV | **63 mV** | **0,77** |
| Écart de tension (99e percentile) | 59 mV | **235 mV** | **0,79** |
| Écart de tension maximal | 145 mV | 283 mV | 0,65 |
| Température maximale | 45,1 °C | 41,9 °C | 0,23 |

> AUROC : 0,5 = hasard, 1 = séparation parfaite. Sur les 32 véhicules de test seuls : 0,84 (écart moyen), 0,83 (99e percentile).

**Par marque (écart moyen par véhicule) :**

| Marque | Véhicules | En panne | Normaux | En panne | AUROC |
|---|---|---|---|---|---|
| 1 | 198 | 30 | 20 mV | 63 mV | 0,77 |
| 2 | 49 | 16 | 18 mV | 55 mV | **0,90** |
| 3 | 100 | 9 | 28 mV | 31 mV | 0,57 |

→ Ça marche très bien sur les marques 1 et 2, **pas sur la marque 3** (ses pannes ne sont pas des déséquilibres de cellule). Total : 700 582 extraits, 347 véhicules, 55 en panne (conforme à l'article).

**Une règle simple sur l'écart moyen par véhicule (marque 1) :**

| Règle | Véhicules en panne détectés | Véhicules normaux signalés à tort |
|---|---|---|
| Écart moyen > 30 mV | 17 / 30 (57 %) | 8 / 168 (5 %) |
| Écart moyen > 40 mV | 16 / 30 (53 %) | 5 / 168 (3 %) |
| Écart **maximal** > 60 mV (pour comparer) | 18 / 30 (60 %) | **75 / 168 (45 %)** |

**Ce que ça montre :**
1. ✅ **L'écart entre cellules détecte bien de vraies pannes** → notre indicateur « cellule » est validé : environ la moitié des véhicules en panne, avec très peu de fausses alertes.
2. ✅ Il faut regarder l'écart **moyen (ou au 99e percentile)**, pas le maximum : avec le maximum, près d'un véhicule normal sur deux est signalé (un pic isolé n'est pas une panne). → **À appliquer aussi aux BAIC** : alerte sur `ecart_v_moyen` plutôt que sur `ecart_v_max`.
3. ⚠️ Environ **13 véhicules en panne sur 30 n'ont aucun déséquilibre de cellule** : leur panne est d'un autre type. L'écart de tension ne détecte pas tout — d'où l'intérêt de combiner plusieurs approches (vote).
4. ❌ **La température ne sépare pas** les véhicules en panne → Zhang n'aide pas pour la surchauffe (on garde 45/48 °C sur les BAIC).
5. ⚠️ Les **seuils ne se transfèrent pas** tels quels (écart normal ≈ 20 mV chez Zhang, médiane 41 mV par session chez nous) → on ne fusionne pas les deux datasets.
6. ⚠️ Étiquettes **par véhicule** (un véhicule en panne a aussi des extraits normaux) et véhicules en panne avec ≈ 6× moins d'extraits : à mentionner comme limite.

**Concrètement, l'option D =**

| | Surchauffe | Cellule |
|---|---|---|
| Données BAIC | Seuils 45/48 °C + 4 approches + vote (inchangé) | Option C : alerte 60 mV / critique 100 mV, pas de modèle 2, véhicule 9 = résultat |
| Données Zhang | — | **Validation** : rappel/AUROC de notre règle + petit classifieur entraîné sur les vraies étiquettes |

- ✅ Respecte le rapport (9 étapes, 4 approches, vote).
- ✅ Répond honnêtement à « sans étiquettes de panne » : on **valide** nos indicateurs sur de vraies pannes publiées.
- ✅ Argument fort devant le jury.
- ⚠️ Un peu plus de travail (voir tableau ci-dessous).

**Travail en plus si l'option D est retenue :**

| Qui | Tâche |
|---|---|
| Kevin | `src/external_zhang.py` : lecture **sécurisée** des `.pkl` (jamais `pickle.load` / `torch.load` directement, unpickler restreint à numpy) → un parquet d'indicateurs par extrait |
| Ivan Jr | `src/models/cellule_zhang.py` : évaluer la règle 60/100 mV + classifieur (Random Forest / XGBoost) sur les vraies étiquettes, séparation par véhicule |
| Mohamed | Une page « validation sur vraies pannes » dans le dashboard |
| Stevy | Suivre l'expérience dans MLflow |
| Tous | Données dans `data/external/` (**non versionné**, 7 Go) — chacun télécharge depuis figshare |

## 5. Ce qui change si l'option C est retenue

| Qui | Changement |
|---|---|
| Tous | `config.py` : ajouter `ECART_TENSION_ALERTE = 0.06` (à côté de `ECART_TENSION_MAX = 0.1`) |
| Mohamed | Vote des seuils à 2 niveaux pour la cellule · fusion à 3 votes pour la cellule · page « véhicule 9 » dans le dashboard |
| Ivan Jr | Modèle 2 seulement pour la surchauffe : **8 entraînements au lieu de 10** |
| Stevy | Isolation Forest : vérifier qu'il repère bien le véhicule 9 |
| Kevin | Vérifier les 16 sessions de plus de 12 h (jusqu'à 65 h) |
| Rapport | Corriger « 16 mV » · justifier les deux niveaux · expliquer le retrait du modèle 2 cellule |

## 6. Vote

Chacun écrit son choix (A, B, C ou D) et un commentaire s'il le souhaite, puis fait un commit.

| Membre | Choix | Commentaire |
|---|---|---|
| Mobio Ivan Junior | **D** | Valide notre indicateur cellule sur de vraies pannes (Zhang et al.) |
| Traore Mohamed | | |
| Kabre Yann Kevin | | |
| Gatsoundou Junior Stevy | | |

**Décision finale :** _à compléter_
