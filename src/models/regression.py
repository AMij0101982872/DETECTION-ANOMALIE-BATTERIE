"""Modèle 1 — Régression : « elle chauffe plus que prévu » (membre 3, Ivan Jr).

Idée : on prédit ce qu'une session *devrait* atteindre compte tenu de ses conditions
(courant, SOC de départ, température de départ, mois, durée). Une session dont la valeur
réelle dépasse nettement la valeur prévue est anormale, même sous le seuil absolu.

- Cibles : temp_max (surchauffe) et ecart_v_max (cellule).
- Algorithmes : régression linéaire (référence), Random Forest, XGBoost → 6 entraînements.
- Évaluation : séparation par véhicule (config.TEST_VEHICLES), MAE / RMSE / R².
- Choix du meilleur algorithme : validation croisée **par véhicule** sur le train
  (GroupKFold), jamais sur le test.
- Vote : résidu (réel − prévu) > 99e percentile des résidus hors-échantillon du train.
  On utilise des résidus hors-échantillon : sur ses propres données d'entraînement, une
  forêt « connaît » les réponses et ses résidus seraient trop petits.

Utilisation : python -m src.models.regression
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, SESSIONS_PARQUET, TEST_VEHICLES

FEATURES = ["courant_moyen", "soc_debut", "temp_debut", "mois", "duree_min"]
CIBLES = {"surchauffe": "temp_max", "cellule": "ecart_v_max"}
PERCENTILE_VOTE = 99      # résidu au-delà du 99e percentile → vote anomalie
NB_PLIS = 4               # validation croisée par véhicule
VOTES_PARQUET = PROCESSED_DIR / "votes_regression.parquet"


def algorithmes():
    """Les 3 algorithmes comparés (nouvelles instances à chaque appel)."""
    return {
        "lineaire": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                               random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                subsample=0.8, n_jobs=-1, random_state=RANDOM_STATE),
    }


def separer(sessions):
    """Train = 16 véhicules, test = config.TEST_VEHICLES (jamais vus à l'entraînement)."""
    test = sessions["vehicle_id"].isin(TEST_VEHICLES)
    return sessions[~test], sessions[test]


def metriques(y_reel, y_prevu):
    return {"mae": mean_absolute_error(y_reel, y_prevu),
            "rmse": float(np.sqrt(mean_squared_error(y_reel, y_prevu))),
            "r2": r2_score(y_reel, y_prevu)}


def entrainer(sessions, journaliser=True, verbose=True):
    """Entraîne les 6 modèles et choisit le meilleur par cible.

    Renvoie (resultats, modeles, seuils) :
    - resultats : une ligne par (cible, algorithme) avec MAE/RMSE/R² en validation croisée
      (train) et sur le test, et le seuil de vote
    - modeles   : {anomalie: meilleur modèle, réentraîné sur tout le train}
    - seuils    : {anomalie: seuil de résidu au-delà duquel on vote anomalie}
    """
    train, test = separer(sessions)
    groupes = train["vehicle_id"]
    plis = GroupKFold(n_splits=min(NB_PLIS, groupes.nunique()))
    lignes, modeles, seuils = [], {}, {}

    for anomalie, cible in CIBLES.items():
        candidats = {}
        for nom, modele in algorithmes().items():
            # Prédictions hors-échantillon sur le train : choix du modèle + seuil de vote
            oof = cross_val_predict(modele, train[FEATURES], train[cible], groups=groupes, cv=plis)
            seuil = float(np.percentile(train[cible] - oof, PERCENTILE_VOTE))
            modele.fit(train[FEATURES], train[cible])
            prevu = modele.predict(test[FEATURES])
            m_cv, m_test = metriques(train[cible], oof), metriques(test[cible], prevu)
            taux_vote_test = float(((test[cible] - prevu) > seuil).mean())
            lignes.append({"anomalie": anomalie, "cible": cible, "algorithme": nom,
                           **{f"{k}_cv": v for k, v in m_cv.items()},
                           **{f"{k}_test": v for k, v in m_test.items()},
                           "seuil_residu": seuil, "taux_vote_test": taux_vote_test})
            candidats[nom] = (m_cv["mae"], modele, seuil)
            if verbose:
                print(f"  {cible:<12} {nom:<14} MAE cv {m_cv['mae']:.4f} | test {m_test['mae']:.4f}"
                      f"  R² test {m_test['r2']:.3f}", flush=True)
            if journaliser:
                _journaliser(anomalie, cible, nom, modele, m_cv, m_test, seuil, test[FEATURES])

        meilleur = min(candidats, key=lambda n: candidats[n][0])
        _, modeles[anomalie], seuils[anomalie] = candidats[meilleur]

    resultats = pd.DataFrame(lignes)
    resultats["meilleur"] = resultats.groupby("anomalie")["mae_cv"].transform("min") == resultats["mae_cv"]
    return resultats, modeles, seuils


def vote_sessions(sessions, modeles, seuils):
    """Ajoute les résidus et les votes (0/1) : un vote par anomalie + vote_regression (l'un ou l'autre).

    vote_regression est gardé pour la fusion actuelle (fusion.py) ; avec le vote par anomalie
    (option D), Mohamed utilise vote_regression_surchauffe et vote_regression_cellule.
    """
    sortie = sessions[["vehicle_id", "session_id"]].copy()
    for anomalie, cible in CIBLES.items():
        prevu = modeles[anomalie].predict(sessions[FEATURES])
        sortie[f"{cible}_prevu"] = prevu
        sortie[f"residu_{cible}"] = sessions[cible].to_numpy() - prevu
        sortie[f"vote_regression_{anomalie}"] = (sortie[f"residu_{cible}"] > seuils[anomalie]).astype(int)
    sortie["vote_regression"] = sortie[[f"vote_regression_{a}" for a in CIBLES]].max(axis=1)
    return sortie


def importance_variables(modeles):
    """Importance des variables du meilleur modèle de chaque cible (si c'est un modèle à arbres)."""
    return pd.DataFrame({a: getattr(m, "feature_importances_", np.full(len(FEATURES), np.nan))
                         for a, m in modeles.items()}, index=FEATURES)


def _journaliser(anomalie, cible, nom, modele, m_cv, m_test, seuil, exemple):
    """Enregistre le run dans MLflow via src.mlops.log_run (Stevy), si disponible."""
    try:
        from src.mlops import log_run
    except ImportError:          # mlflow non installé ou src.mlops pas encore sur main
        return None
    return log_run(
        experience="regression", nom_run=f"{nom}_{cible}",
        params={"cible": cible, "algorithme": nom, "features": FEATURES,
                "percentile_vote": PERCENTILE_VOTE, "test_vehicles": TEST_VEHICLES},
        metriques={**{f"{k}_cv": v for k, v in m_cv.items()},
                   **{f"{k}_test": v for k, v in m_test.items()}, "seuil_residu": seuil},
        modele=modele, exemple_entree=exemple.head(5), tags={"membre": "Ivan Jr", "anomalie": anomalie},
    )


def _figure_importance(importance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ax = importance.plot.barh(figsize=(7, 4))
    ax.set_xlabel("Importance")
    ax.set_title("Modèle 1 : importance des variables (meilleur modèle par cible)")
    plt.tight_layout()
    chemin = FIGURES_DIR / "regression_importance.png"
    plt.savefig(chemin, dpi=120)
    plt.close()
    return chemin


if __name__ == "__main__":
    sessions = pd.read_parquet(SESSIONS_PARQUET)
    print(f"{len(sessions):,} sessions | test : véhicules {TEST_VEHICLES}")
    resultats, modeles, seuils = entrainer(sessions)

    pd.set_option("display.width", 200)
    print("\n=== Comparaison des 6 modèles ===")
    print(resultats.drop(columns=["anomalie"]).round(4).to_string(index=False))

    votes = vote_sessions(sessions, modeles, seuils)
    votes.to_parquet(VOTES_PARQUET, index=False)
    print(f"\nVotes enregistrés : {VOTES_PARQUET}")
    test = votes["vehicle_id"].isin(TEST_VEHICLES)
    for a in CIBLES:
        col = f"vote_regression_{a}"
        print(f"  {col:<28} train {votes.loc[~test, col].mean():.2%} | test {votes.loc[test, col].mean():.2%}")

    importance = importance_variables(modeles)
    print("\n=== Importance des variables ===")
    print(importance.round(3))
    print(f"Figure : {_figure_importance(importance)}")
