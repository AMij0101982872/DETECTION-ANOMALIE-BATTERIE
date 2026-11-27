"""Modèle 2 — Alerte précoce : « ça va arriver » (membre 3, Ivan Jr).

Question : en ne regardant que les 15 premières minutes d'une session de charge, peut-on
prévoir qu'elle finira en surchauffe (temp_max > 45 °C) ?

- Surchauffe uniquement (option D : pas de modèle 2 pour la cellule, 90 % des cas = véhicule 9).
- Population : sessions de plus de 15 min dont la température n'a pas encore dépassé 45 °C
  à la 15e minute (sinon il n'y a plus rien à anticiper : ces sessions votent 1 directement).
- Entrées : uniquement ce qui est connu à la 15e minute (colonnes *_15min + conditions de départ).
- Algorithmes : régression logistique, Random Forest et XGBoost en classification, classes
  rééquilibrées (≈ 1 % de positifs) ; une règle simple sur temp_15min sert de référence.
  Retenu : la régression logistique (meilleure PR-AUC en validation croisée, la plus simple).
- Choix du modèle (PR-AUC) et du seuil de probabilité (F2 : le rappel compte double) par
  validation croisée par véhicule sur le train, jamais sur le test.

Utilisation : python -m src.models.alerte_precoce
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier

from src.config import PROCESSED_DIR, RANDOM_STATE, SESSIONS_PARQUET, TEMP_ALERTE, TEST_VEHICLES

FEATURES = ["temp_15min", "vitesse_temp_15min", "courant_moyen_15min", "ecart_v_max_15min",
            "soc_15min", "temp_debut", "soc_debut", "mois"]
CIBLE = "label_surchauffe"
NB_PLIS = 4
BETA = 2                  # F2 : le rappel pèse deux fois plus que la précision
VOTES_PARQUET = PROCESSED_DIR / "votes_alerte_precoce.parquet"


def ajouter_label(sessions):
    """label_surchauffe = 1 si la session finit au-delà de TEMP_ALERTE (45 °C)."""
    sessions = sessions.copy()
    sessions[CIBLE] = (sessions["temp_max"] > TEMP_ALERTE).astype(int)
    return sessions


def masques(sessions):
    """Trois groupes de sessions :
    - evaluable : plus de 15 min, pas encore en surchauffe à 15 min → le modèle décide
    - deja      : déjà au-delà de 45 °C à 15 min → alerte immédiate (vote 1)
    - le reste  : sessions de moins de 15 min → pas de fenêtre de 15 min (vote 0)
    """
    longue = sessions["session_longue"].astype(bool)
    deja = longue & (sessions["temp_15min"] > TEMP_ALERTE)
    return longue & ~deja, deja


def population(sessions):
    """Sessions sur lesquelles le modèle est entraîné et évalué, avec leur étiquette."""
    evaluable, _ = masques(sessions)
    return ajouter_label(sessions[evaluable])


def separer(pop):
    test = pop["vehicle_id"].isin(TEST_VEHICLES)
    return pop[~test], pop[test]


def algorithmes(y_train):
    """Régression logistique, Random Forest et XGBoost, rééquilibrés pour ≈ 1 % de positifs.

    Hyperparamètres retenus par recherche en grille (GroupKFold, PR-AUC), voir le notebook 05 :
    régression logistique C = 1 ; XGBoost max_depth = 3, n_estimators = 200.
    """
    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    return {
        "regression_logistique": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced_subsample", n_jobs=-1,
                                                random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 scale_pos_weight=ratio, eval_metric="aucpr", n_jobs=-1,
                                 random_state=RANDOM_STATE),
    }


def seuil_f2(y, proba):
    """Seuil de probabilité qui maximise le F2 (rappel prioritaire)."""
    precision, rappel, seuils = precision_recall_curve(y, proba)
    f2 = (1 + BETA**2) * precision * rappel / np.maximum(BETA**2 * precision + rappel, 1e-12)
    return float(seuils[np.argmax(f2[:-1])])


def metriques(y, proba, seuil):
    pred = (proba >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, pred, zero_division=0), "rappel": recall_score(y, pred),
            "f1": f1_score(y, pred), "f2": fbeta_score(y, pred, beta=BETA),
            "pr_auc": average_precision_score(y, proba), "roc_auc": roc_auc_score(y, proba),
            "vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn)}


def entrainer(sessions, journaliser=True, verbose=True):
    """Compare la règle de référence, Random Forest et XGBoost ; renvoie (resultats, modele, seuil).

    - resultats : une ligne par approche, métriques en validation croisée (train) et sur le test
    - modele    : le meilleur (PR-AUC en validation croisée), réentraîné sur tout le train
    - seuil     : seuil de probabilité choisi par F2 sur les prédictions hors-échantillon
    """
    train, test = separer(population(sessions))
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[CIBLE], test[FEATURES], test[CIBLE]
    plis = GroupKFold(n_splits=NB_PLIS)
    lignes, candidats = [], {}

    # Référence : « temp_15min » utilisée comme score (plus c'est chaud à 15 min, plus c'est risqué)
    s_ref = seuil_f2(y_tr, train["temp_15min"])
    lignes.append({"approche": "regle_temp_15min", **_prefixer(metriques(y_tr, train["temp_15min"], s_ref), "cv"),
                   **_prefixer(metriques(y_te, test["temp_15min"], s_ref), "test"), "seuil": s_ref})

    for nom, modele in algorithmes(y_tr).items():
        oof = cross_val_predict(modele, X_tr, y_tr, groups=train["vehicle_id"], cv=plis,
                                method="predict_proba")[:, 1]
        seuil = seuil_f2(y_tr, oof)
        modele.fit(X_tr, y_tr)
        proba_te = modele.predict_proba(X_te)[:, 1]
        m_cv, m_te = metriques(y_tr, oof, seuil), metriques(y_te, proba_te, seuil)
        lignes.append({"approche": nom, **_prefixer(m_cv, "cv"), **_prefixer(m_te, "test"), "seuil": seuil})
        candidats[nom] = (m_cv["pr_auc"], modele, seuil)
        if verbose:
            print(f"  {nom:<14} PR-AUC cv {m_cv['pr_auc']:.3f} | test {m_te['pr_auc']:.3f}  "
                  f"rappel test {m_te['rappel']:.2f}  précision test {m_te['precision']:.2f}", flush=True)
        if journaliser:
            _journaliser(nom, modele, m_cv, m_te, seuil, X_te)

    meilleur = max(candidats, key=lambda n: candidats[n][0])
    resultats = pd.DataFrame(lignes)
    resultats["meilleur"] = resultats["approche"] == meilleur
    _, modele, seuil = candidats[meilleur]
    return resultats, modele, seuil


def _prefixer(d, suffixe):
    return {f"{k}_{suffixe}": v for k, v in d.items()}


def vote_sessions(sessions, modele, seuil):
    """Vote pour toutes les sessions (contrat TACHE.md : votes_alerte_precoce.parquet).

    - evaluable : vote 1 si la probabilité ≥ seuil
    - déjà > 45 °C à 15 min : vote 1 (alerte immédiate)
    - moins de 15 min : vote 0, ap_evaluable = faux
    """
    evaluable, deja = masques(sessions)
    sortie = sessions[["vehicle_id", "session_id"]].copy()
    sortie["proba_alerte_precoce"] = np.nan
    sortie.loc[evaluable, "proba_alerte_precoce"] = modele.predict_proba(sessions.loc[evaluable, FEATURES])[:, 1]
    sortie["vote_alerte_precoce_surchauffe"] = ((sortie["proba_alerte_precoce"] >= seuil) | deja).astype(int)
    sortie["ap_evaluable"] = evaluable | deja
    sortie["raison_alerte_precoce"] = np.select(
        [deja, sortie["vote_alerte_precoce_surchauffe"].eq(1), ~(evaluable | deja)],
        ["déjà > 45 °C à 15 min", "risque de surchauffe prévu", "session < 15 min"], default="")
    return sortie


def minutes_avant_surchauffe(sessions):
    """Pour chaque session, minutes entre son début et la 1re mesure > 45 °C (NaN si jamais).

    Relit les mesures nettoyées (étape 3) et reconstruit les mêmes session_id que l'étape 5.
    L'alerte tombe à la 15e minute : l'avance gagnée = cette valeur − 15.
    """
    from src.config import CLEAN_PARQUET
    from src.sessions import ajouter_session_id

    mesures = ajouter_session_id(pd.read_parquet(CLEAN_PARQUET, columns=["vehicle_id", "time", "temp_max"]))
    premiere = mesures[mesures["temp_max"] > TEMP_ALERTE].groupby("session_id")["time"].min()
    debut = sessions.set_index("session_id")["debut"]
    return ((premiere.reindex(debut.index) - debut).dt.total_seconds() / 60).rename("minutes_avant_45")


def importance_variables(modele):
    """Importance des arbres, ou |coefficient| standardisé pour la régression logistique."""
    if hasattr(modele, "feature_importances_"):
        valeurs = modele.feature_importances_
    else:
        valeurs = np.abs(modele[-1].coef_[0])
    return pd.Series(valeurs, index=FEATURES).sort_values(ascending=False)


def _journaliser(nom, modele, m_cv, m_te, seuil, exemple):
    """Enregistre le run dans MLflow via src.mlops.log_run (Stevy), si mlflow est installé."""
    try:
        from src.mlops import log_run
    except ImportError:
        return None
    return log_run(
        experience="alerte_precoce", nom_run=f"{nom}_surchauffe",
        params={"algorithme": nom, "features": FEATURES, "cible": CIBLE, "beta": BETA,
                "seuil_temperature": TEMP_ALERTE, "test_vehicles": TEST_VEHICLES},
        metriques={**_prefixer({k: v for k, v in m_cv.items()}, "cv"),
                   **_prefixer({k: v for k, v in m_te.items()}, "test"), "seuil_proba": seuil},
        modele=modele, exemple_entree=exemple.head(5), tags={"membre": "Ivan Jr", "anomalie": "surchauffe"},
    )


if __name__ == "__main__":
    sessions = pd.read_parquet(SESSIONS_PARQUET)
    pop = population(sessions)
    print(f"{len(pop):,} sessions évaluables, {int(pop[CIBLE].sum())} finissent > {TEMP_ALERTE} °C "
          f"({pop[CIBLE].mean():.2%}) | test : véhicules {TEST_VEHICLES}")
    resultats, modele, seuil = entrainer(sessions)

    pd.set_option("display.width", 220)
    colonnes = ["approche", "pr_auc_cv", "pr_auc_test", "rappel_test", "precision_test", "f2_test",
                "vp_test", "fn_test", "fp_test", "seuil", "meilleur"]
    print("\n=== Comparaison ===")
    print(resultats[colonnes].round(3).to_string(index=False))

    votes = vote_sessions(sessions, modele, seuil)
    votes.to_parquet(VOTES_PARQUET, index=False)
    print(f"\nVotes : {VOTES_PARQUET} ({int(votes['vote_alerte_precoce_surchauffe'].sum())} sessions signalées)")
    print("\n=== Importance des variables ===")
    print(importance_variables(modele).round(3).to_string())
