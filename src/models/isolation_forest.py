"""Modèle 3 — Atypique (non supervisé) : « cette charge est bizarre ».

Membre 4 (Stevy). Démarche complète, hypothèses et résultats :
notebooks/04_modele3_isolation_forest.ipynb (étapes 1 à 5).

Modèle : **Isolation Forest unilatéral** (Isolation Forest avec écrêtage unilatéral à la
médiane). Pipeline : imputation (médiane) → écrêtage bas à la médiane → normalisation →
Isolation Forest. L'écrêtage remplace toute valeur inférieure à la médiane d'entraînement par
la médiane : seules les déviations vers le HAUT (les seules dangereuses) isolent une session.

Décisions (notebook) :

| Choix                                     | Raison (étape du notebook)                                   |
|-------------------------------------------|--------------------------------------------------------------|
| temp_max, vitesse_temp, ecart_v_max       | 10 colonnes diluent le signal ; ecart_v_moyen est redondant (2) |
| contamination = 2 %                       | +30 points de rappel de 1 à 2 %, +1,7 de 2 à 3 % (2)          |
| sessions < 5 min non évaluables (vote 0)  | artefacts : 1,1 % des sessions mais ≈ 15 % des votes (1, 2)   |
| écrêtage unilatéral                       | 38 % des votes venaient de valeurs basses sans danger (4.1)  |
| entraînement sur les véhicules hors test  | séparation par véhicule (règle du projet)                    |

Résultats (sessions normales des véhicules de test, anomalies injectées) : voir le notebook,
étape 5. Limite connue : un véhicule « différent de naissance » (n° 18, écart de tension haut
mais stable) est plus souvent signalé, car chaque session est comparée à toute la flotte.

Sortie pour la fusion (contrat TACHE.md) : data/processed/votes_isolation_forest.parquet
  vehicle_id, session_id, vote_isolation_forest (0/1), if_evaluable (bool),
  score_isolation_forest (> 0 = atypique), raison_isolation_forest (si vote = 1)

Lancer :  python -m src.models.isolation_forest            (après python main.py)
          python -m src.models.isolation_forest --sans-mlflow
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (PROCESSED_DIR, RANDOM_STATE, SESSIONS_PARQUET, TEMP_ALERTE,
                        TEST_VEHICLES)

FEATURES = ["temp_max", "vitesse_temp", "ecart_v_max"]
CONTAMINATION = 0.02
N_ESTIMATORS = 300
DUREE_MIN_EVALUABLE = 5          # minutes
VOTES_PARQUET = PROCESSED_DIR / "votes_isolation_forest.parquet"

# Seuil d'alerte cellule de l'option D (DECISION_SEUIL_CELLULE.md). À remplacer par
# config.ECART_TENSION_ALERTE quand Mohamed l'aura ajouté.
ECART_TENSION_ALERTE = 0.060     # V

# Explication d'un vote : colonne la plus au-dessus de la normale -> raison affichée
RAISONS = {"ecart_v_max": "cellule", "temp_max": "température", "vitesse_temp": "montée rapide"}

# Anomalies simulées pour l'évaluation. Cellule « réaliste » : l'écart moyen monte de
# RATIO_ECART_MOYEN × la hausse de l'écart max (dégradation du véhicule 9, notebook 2.1).
INJECTION = {
    "surchauffe": {"temp_max": 8.0},        # + 8 °C
    "cellule": {"ecart_v_max": 0.060},      # + 60 mV (les écarts sont en volts)
}
RATIO_ECART_MOYEN = 0.33


class EcretageBas(BaseEstimator, TransformerMixin):
    """Remplace les valeurs inférieures à la médiane d'entraînement par cette médiane.

    Retient aussi l'écart interquartile, utilisé pour expliquer les votes (z-score robuste).
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.medianes_ = np.nanmedian(X, axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.iqr_ = np.where(q3 > q1, q3 - q1, 1.0)
        return self

    def transform(self, X):
        return np.maximum(np.asarray(X, dtype=float), self.medianes_)


def evaluable(sessions):
    """Session assez longue pour que ses features soient fiables (sinon pas de vote)."""
    return sessions["duree_min"] >= DUREE_MIN_EVALUABLE


def _x(sessions):
    # inf possible sur vitesse_temp si durée nulle : traité comme une valeur manquante
    return sessions[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)


def entrainer(sessions, contamination=CONTAMINATION, n_estimators=N_ESTIMATORS):
    """Entraîne le pipeline sur les sessions évaluables des véhicules d'entraînement."""
    train = sessions[~sessions["vehicle_id"].isin(TEST_VEHICLES) & evaluable(sessions)]
    modele = Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("ecretage_bas", EcretageBas()),
        ("normalisation", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    modele.fit(_x(train))
    return modele


def score_sessions(sessions, modele):
    """Score de bizarrerie : > 0 = plus atypique que le seuil de contamination."""
    return pd.Series(-modele.decision_function(_x(sessions)), index=sessions.index,
                     name="score_isolation_forest")


def vote_sessions(sessions, modele):
    """1 si la session est atypique ; 0 sinon, et toujours 0 si elle n'est pas évaluable."""
    atypique = modele.predict(_x(sessions)) == -1
    return pd.Series((atypique & evaluable(sessions).to_numpy()).astype(int),
                     index=sessions.index, name="vote_isolation_forest")


def expliquer(sessions, modele):
    """Raison de chaque vote : la colonne la plus au-dessus de sa médiane (z-score robuste)."""
    ecretage = modele.named_steps["ecretage_bas"]
    z = (_x(sessions) - ecretage.medianes_) / ecretage.iqr_
    dominante = z.fillna(-np.inf).idxmax(axis=1)
    return dominante.map(RAISONS).rename("raison_isolation_forest")


def table_votes(sessions, modele):
    """Fichier de votes pour la fusion (une ligne par session, contrat TACHE.md)."""
    votes = vote_sessions(sessions, modele)
    return pd.DataFrame({
        "vehicle_id": sessions["vehicle_id"],
        "session_id": sessions["session_id"],
        "vote_isolation_forest": votes,
        "if_evaluable": evaluable(sessions),
        "score_isolation_forest": score_sessions(sessions, modele),
        "raison_isolation_forest": expliquer(sessions, modele).where(votes == 1),
    }).reset_index(drop=True)


def sessions_normales_test(sessions):
    """Sessions évaluables des véhicules de test, sous les deux seuils d'alerte."""
    return sessions[sessions["vehicle_id"].isin(TEST_VEHICLES) & evaluable(sessions)
                    & (sessions["temp_max"] < TEMP_ALERTE)
                    & (sessions["ecart_v_max"] <= ECART_TENSION_ALERTE)]


def injecter_anomalies(sessions, type_anomalie, n=None, random_state=RANDOM_STATE, taille=None):
    """Copie des sessions avec une anomalie simulée (n=None : toutes les sessions).

    surchauffe : temp_max + taille (°C), vitesse_temp recalculée ;
    cellule    : ecart_v_max + taille (V) et ecart_v_moyen + RATIO_ECART_MOYEN × taille.
    """
    if type_anomalie not in INJECTION:
        raise ValueError(f"type_anomalie doit être parmi {list(INJECTION)}")
    colonne, defaut = next(iter(INJECTION[type_anomalie].items()))
    taille = defaut if taille is None else taille
    out = sessions if n is None else sessions.sample(n=min(n, len(sessions)), random_state=random_state)
    out = out.copy()
    out[colonne] = out[colonne] + taille
    if type_anomalie == "surchauffe":
        duree = out["duree_min"].where(out["duree_min"] > 0)
        out["vitesse_temp"] = (out["temp_max"] - out["temp_debut"]) / duree
    else:
        out["ecart_v_moyen"] = out["ecart_v_moyen"] + RATIO_ECART_MOYEN * taille
    return out


def evaluer_injection(sessions, modele, n=None, random_state=RANDOM_STATE):
    """Évalue le modèle sur des anomalies injectées dans les sessions normales du test.

    - rappel_<type> : part des sessions modifiées qui reçoivent le vote 1
    - auroc_<type>  : le score sépare-t-il les sessions modifiées de ces mêmes sessions avant
      modification ? (0,5 = hasard, 1 = parfait)
    - fausses_alarmes : part des sessions normales du test signalées
    """
    normales = sessions_normales_test(sessions)
    if n is not None:
        normales = normales.sample(n=min(n, len(normales)), random_state=random_state)
    resultats = {"fausses_alarmes": float(vote_sessions(normales, modele).mean())}
    s_normales = score_sessions(normales, modele)
    for type_anomalie in INJECTION:
        injectees = injecter_anomalies(normales, type_anomalie)
        resultats[f"rappel_{type_anomalie}"] = float(vote_sessions(injectees, modele).mean())
        y = np.r_[np.zeros(len(normales)), np.ones(len(injectees))]
        scores = np.r_[s_normales, score_sessions(injectees, modele)]
        resultats[f"auroc_{type_anomalie}"] = float(roc_auc_score(y, scores))
    return resultats


def taux_par_vehicule(sessions, modele):
    """Part de sessions atypiques par véhicule, parmi les sessions évaluables."""
    ok = evaluable(sessions)
    votes = vote_sessions(sessions[ok], modele)
    return votes.groupby(sessions.loc[ok, "vehicle_id"]).mean().sort_values(ascending=False)


def enregistrer_mlflow(modele, sessions, metriques, artefacts=None):
    """Enregistre le run dans MLflow et inscrit le modèle au Model Registry."""
    from src.mlops import log_run
    return log_run(
        experience="isolation_forest",
        nom_run="isolation_forest_unilateral",
        params={"modele": "Isolation Forest unilatéral (écrêtage bas à la médiane)",
                "features": FEATURES, "contamination": CONTAMINATION,
                "n_estimators": N_ESTIMATORS, "duree_min_evaluable": DUREE_MIN_EVALUABLE,
                "test_vehicles": TEST_VEHICLES, "random_state": RANDOM_STATE},
        metriques=metriques,
        modele=modele,
        exemple_entree=_x(sessions).head(5),
        artefacts=artefacts,
        tags={"membre": "Stevy", "modele": "3 - atypique",
              "notebook": "notebooks/04_modele3_isolation_forest.ipynb"},
        nom_registry="isolation_forest",
    )


def main(suivi_mlflow=True):
    sessions = pd.read_parquet(SESSIONS_PARQUET)
    print(f"{len(sessions):,} sessions, {sessions['vehicle_id'].nunique()} véhicules")
    modele = entrainer(sessions)

    votes = table_votes(sessions, modele)
    votes.to_parquet(VOTES_PARQUET, index=False)
    print(f"Votes : {VOTES_PARQUET} ({int(votes['vote_isolation_forest'].sum())} sessions "
          f"signalées, {int((~votes['if_evaluable']).sum())} non évaluables)")

    resultats = evaluer_injection(sessions, modele)
    par_vehicule = taux_par_vehicule(sessions, modele)
    resultats["rang_vehicule_9"] = int(par_vehicule.index.get_loc(9)) + 1
    resultats["taux_signale"] = float(votes.loc[votes["if_evaluable"], "vote_isolation_forest"].mean())
    for cle, valeur in resultats.items():
        print(f"  {cle:20s} {valeur:.3f}")
    print("  top 5 :", ", ".join(f"#{v} {t:.1%}" for v, t in par_vehicule.head(5).items()))

    if suivi_mlflow:
        run_id = enregistrer_mlflow(modele, sessions, resultats)
        print(f"MLflow run : {run_id} (modèle inscrit sous « isolation_forest »)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sans-mlflow", action="store_true")
    args = parser.parse_args()
    main(suivi_mlflow=not args.sans_mlflow)
