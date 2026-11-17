"""Modèle 3 — Atypique (non supervisé, Isolation Forest) : « cette charge est bizarre ».

Membre 4 (Stevy).

- Entrée : les features de session liées aux deux anomalies, normalisées (StandardScaler),
  sans étiquette.
- Sortie : un score de bizarrerie par session (plus il est grand, plus la session est atypique)
  et un vote 0/1 (1 = session atypique).
- Entraînement uniquement sur les véhicules d'entraînement (config.TEST_VEHICLES exclus).
- Évaluation (§3.7) : on injecte de fausses anomalies dans des sessions normales de test
  (surchauffe : temp_max + 8 °C ; cellule : ecart_v_max + 60 mV) et on mesure la part retrouvée.

Choix des features et de `contamination` (essais sur les 31 035 sessions BAIC, 17/11/2026) :

| Features                               | contam. | rappel surch. | rappel cellule | AUROC s / c | rang véh. 9 |
|----------------------------------------|---------|---------------|----------------|-------------|-------------|
| 10 features de session complète        | 1 %     | 3 %           | 2 %            | 0,87 / 0,86 | 12 / 20     |
| 10 features de session complète        | 2 %     | 2 %           | 1 %            | 0,91 / 0,89 | 15 / 20     |
| 7 thermiques + cellule + courant       | 2 %     | 12 %          | 8 %            | 0,95 / 0,93 | 11 / 20     |
| **4 : temp_max, vitesse_temp, écarts** | 1 %     | 35 %          | 25 %           | 0,97 / 0,99 | 5 / 20      |
| **4 : temp_max, vitesse_temp, écarts** | **2 %** | **64 %**      | **58 %**       | 0,98 / 1,00 | **2 / 20**  |

(300 arbres ; 200 anomalies injectées par type dans des sessions des véhicules 17 à 20.
À 2 %, le véhicule le plus signalé est le n° 8 (5,8 % de ses sessions), devant le n° 9 (3,8 %).)

- Les colonnes de contexte (durée, SOC, courant) diluent le signal : Isolation Forest tire au
  hasard la colonne de chaque coupure, donc avec 10 colonnes il coupe rarement sur l'écart
  de tension. On garde les 4 colonnes qui décrivent les deux anomalies.
- 2 % plutôt que 1 % : ≈ 2 × le taux d'anomalies des seuils (≈ 1 % surchauffe + 0,6 %
  cellule) ; le modèle signale ainsi environ 600 sessions, un volume encore vérifiable.
- Limite : l'évaluation injecte des anomalies sur ces mêmes colonnes, ce qui favorise ce
  choix. Le modèle ne verra pas une anomalie qui ne touche que la durée ou le courant.

Comparaison relancée par `python -m src.models.isolation_forest` (1 % et 2 %).

Lancer :  python -m src.models.isolation_forest            (après python main.py)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, SESSIONS_PARQUET, TEST_VEHICLES

# Colonnes de sessions_features.parquet décrivant la surchauffe et la cellule (voir plus haut).
# Les colonnes *_15min servent au modèle 2 et sont vides pour les sessions courtes.
FEATURES = ["temp_max", "vitesse_temp", "ecart_v_moyen", "ecart_v_max"]
CONTAMINATION = 0.02

# Anomalies simulées pour l'évaluation (TACHE.md)
INJECTION = {
    "surchauffe": {"temp_max": 8.0},        # + 8 °C
    "cellule": {"ecart_v_max": 0.060},      # + 60 mV (les écarts sont en volts)
}


def entrainer(sessions, contamination=CONTAMINATION, n_estimators=300):
    """Entraîne imputation + normalisation + Isolation Forest sur les véhicules d'entraînement."""
    train = sessions[~sessions["vehicle_id"].isin(TEST_VEHICLES)]
    modele = Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("normalisation", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    modele.fit(_x(train))
    return modele


def _x(sessions):
    # inf possible sur vitesse_temp si durée nulle : traité comme une valeur manquante
    return sessions[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)


def score_sessions(sessions, modele):
    """Score de bizarrerie : > 0 = plus atypique que le seuil de contamination."""
    return pd.Series(-modele.decision_function(_x(sessions)), index=sessions.index,
                     name="score_if")


def vote_sessions(sessions, modele):
    """1 si la session est classée atypique, sinon 0."""
    return pd.Series((modele.predict(_x(sessions)) == -1).astype(int), index=sessions.index,
                     name="vote_isolation_forest")


def injecter_anomalies(sessions, type_anomalie, n=200, random_state=RANDOM_STATE):
    """Copie n sessions jugées normales par le modèle et y ajoute une anomalie simulée."""
    if type_anomalie not in INJECTION:
        raise ValueError(f"type_anomalie doit être parmi {list(INJECTION)}")
    echantillon = sessions.sample(n=min(n, len(sessions)), random_state=random_state).copy()
    for colonne, delta in INJECTION[type_anomalie].items():
        echantillon[colonne] = echantillon[colonne] + delta
        if colonne == "temp_max":   # garder la vitesse cohérente avec la nouvelle temp_max
            duree = echantillon["duree_min"].where(echantillon["duree_min"] > 0)
            echantillon["vitesse_temp"] = (echantillon["temp_max"] - echantillon["temp_debut"]) / duree
    return echantillon


def evaluer_injection(sessions, modele, n=200, random_state=RANDOM_STATE):
    """Évalue le modèle sur des anomalies simulées dans des sessions de test jugées normales.

    On part des sessions des véhicules de test que le modèle ne signale pas déjà (sinon on
    compterait comme « retrouvée » une session qui était déjà atypique avant l'injection).

    - rappel_<type> : part des sessions modifiées qui reçoivent le vote 1 (dépend de
      `contamination`)
    - auroc_<type>  : le score sépare-t-il les sessions modifiées de ces mêmes sessions avant
      modification ? (indépendant du seuil ; 0,5 = hasard, 1 = parfait)

    Limite connue : Isolation Forest ne coupe qu'à l'intérieur de la plage vue à
    l'entraînement. Une valeur très au-delà du maximum observé reçoit donc à peu près le même
    score que ce maximum, et le score ne grandit pas avec l'ampleur de l'anomalie.
    """
    test = sessions[sessions["vehicle_id"].isin(TEST_VEHICLES)]
    votes_test = vote_sessions(test, modele)
    normales = test[votes_test == 0]
    resultats = {"taux_atypique_test": float(votes_test.mean())}
    for type_anomalie in INJECTION:
        injectees = injecter_anomalies(normales, type_anomalie, n=n, random_state=random_state)
        originales = normales.loc[injectees.index]
        resultats[f"rappel_{type_anomalie}"] = float(vote_sessions(injectees, modele).mean())
        y = np.r_[np.zeros(len(originales)), np.ones(len(injectees))]
        scores = np.r_[score_sessions(originales, modele), score_sessions(injectees, modele)]
        resultats[f"auroc_{type_anomalie}"] = float(roc_auc_score(y, scores))
    return resultats


def taux_par_vehicule(sessions, modele):
    """Part de sessions atypiques par véhicule (le véhicule 9 doit se détacher)."""
    votes = vote_sessions(sessions, modele)
    return votes.groupby(sessions["vehicle_id"]).mean().sort_values(ascending=False)


def main(contaminations=(0.01, CONTAMINATION), suivi_mlflow=True):
    sessions = pd.read_parquet(SESSIONS_PARQUET)
    print(f"{len(sessions):,} sessions, {sessions['vehicle_id'].nunique()} véhicules")
    for contamination in contaminations:
        modele = entrainer(sessions, contamination=contamination)
        resultats = evaluer_injection(sessions, modele)
        par_vehicule = taux_par_vehicule(sessions, modele)
        rang_v9 = int(par_vehicule.index.get_loc(9)) + 1 if 9 in par_vehicule.index else None
        resultats["taux_atypique_vehicule_9"] = float(par_vehicule.get(9, np.nan))
        print(f"\ncontamination = {contamination:.0%}")
        for cle, valeur in resultats.items():
            print(f"  {cle:28s} {valeur:.1%}")
        print(f"  véhicule 9 : rang {rang_v9} / {len(par_vehicule)} (part de sessions atypiques)")
        print("  top 5 :", ", ".join(f"#{v} {t:.1%}" for v, t in par_vehicule.head(5).items()))
        if suivi_mlflow:
            from src.mlops import log_run
            run_id = log_run(
                experience="isolation_forest",
                nom_run=f"isolation_forest_c{contamination}",
                params={"contamination": contamination, "n_estimators": 300,
                        "features": FEATURES, "test_vehicles": TEST_VEHICLES},
                metriques={**resultats, "rang_vehicule_9": rang_v9 or 0},
                modele=modele,
                exemple_entree=_x(sessions).head(5),
                tags={"membre": "Stevy", "modele": "3 - atypique"},
                nom_registry="isolation_forest" if contamination == CONTAMINATION else None,
            )
            print(f"  MLflow run : {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sans-mlflow", action="store_true")
    args = parser.parse_args()
    main(suivi_mlflow=not args.sans_mlflow)
