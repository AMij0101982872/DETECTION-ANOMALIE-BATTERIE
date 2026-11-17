# Données externes : Zhang et al. (2023)

`zhang_extraits.parquet` est un **résumé** du dataset public de Zhang et al. : une ligne par extrait de charge (128 mesures, toutes les 10 s), avec des indicateurs calculés. Les données brutes (7 Go de `.pkl`) ne sont pas dans ce dépôt.

| | |
|---|---|
| Extraits | 700 582 |
| Véhicules | 347 (3 marques), dont **55 en panne confirmée** |
| Colonnes | `marque`, `split`, `car`, `label` (1 = panne, par véhicule), `ecart_moy_mV`, `ecart_max_mV`, `temp_max`, `dtemp`, `courant_moy`, `soc_debut` |

**Calcul des indicateurs (par extrait) :** écart de tension = (`max_single_volt` − `min_single_volt`) × 1000 → moyenne et maximum ; `temp_max` = maximum de `max_temp` ; `dtemp` = max(`max_temp`) − min(`min_temp`) ; `courant_moy` = moyenne du courant (négatif = charge) ; `soc_debut` = SOC de la première mesure.

## Source et licence

Zhang, J., Wang, Y., Jiang, B. et al. *Realistic fault detection of Li-ion battery via dynamical deep learning.* **Nature Communications** 14, 5940 (2023).
Données : figshare, DOI [10.6084/m9.figshare.23659323](https://doi.org/10.6084/m9.figshare.23659323).

Licence : **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. Ce fichier est une version **modifiée** (indicateurs agrégés par extrait, réalisés par l'équipe du projet), redistribuée sous la même licence. Citer la source ci-dessus dans tout travail qui l'utilise.
