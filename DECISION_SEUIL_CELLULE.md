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

Chacun écrit son choix (A, B ou C) et un commentaire s'il le souhaite, puis fait un commit.

| Membre | Choix | Commentaire |
|---|---|---|
| Mobio Ivan Junior | | |
| Traore Mohamed | | |
| Kabre Yann Kevin | | |
| Gatsoundou Junior Stevy | | |

**Décision finale :** _à compléter_
