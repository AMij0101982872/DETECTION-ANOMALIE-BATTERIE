
"""Étape 7 — Suivi des expériences avec MLflow (membre 4).

Utilisation (par tout le monde) :

    from src.mlops import log_run

    run_id = log_run(
        experience="regression",                 # une expérience = un modèle / une question
        nom_run="xgboost_temp_max",
        params={"n_estimators": 300, "cible": "temp_max"},
        metriques={"mae": 0.82, "rmse": 1.10, "r2": 0.91},
        modele=modele_entraine,                  # scikit-learn / XGBoost (optionnel)
        exemple_entree=X_test.head(5),           # pour la signature du modèle (optionnel)
        artefacts=["reports/figures/importance.png"],   # fichiers à joindre (optionnel)
        nom_registry="regression_temp_max",      # pour le Model Registry (optionnel)
    )

Voir les résultats :  mlflow ui --backend-store-uri sqlite:///mlflow.db   → http://127.0.0.1:5000

Le suivi est stocké en local dans `mlflow.db` + `mlartifacts/` à la racine du projet
(non versionnés). Pour un serveur partagé, définir la variable d'environnement
MLFLOW_TRACKING_URI.
"""
import os
from pathlib import Path

import mlflow

from src.config import ROOT_DIR

TRACKING_URI_LOCAL = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
ARTIFACTS_DIR = ROOT_DIR / "mlartifacts"

# MLflow sauvegarde les modèles scikit-learn au format skops (plus sûr que pickle) : il faut
# déclarer les types internes autorisés. Ceux-ci sont ceux des modèles à base d'arbres
# (Random Forest, Isolation Forest...) que nous entraînons nous-mêmes. N'ajouter un type
# qu'après avoir vérifié d'où il vient (jamais la liste complète proposée par l'erreur).
TYPES_SKOPS_CONFIANCE = [
    "sklearn.tree._tree.Tree", "numpy.dtype",
    "src.models.isolation_forest.EcretageBas",   # notre étape d'écrêtage (modèle 3)
]


def configurer_mlflow():
    """Pointe MLflow vers le serveur défini par MLFLOW_TRACKING_URI, sinon vers mlflow.db local."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI_LOCAL)
    mlflow.set_tracking_uri(uri)
    return uri


def _experience(nom):
    """Crée l'expérience si besoin (artefacts dans mlartifacts/ en local) et la sélectionne."""
    if mlflow.get_experiment_by_name(nom) is None:
        emplacement = None
        if mlflow.get_tracking_uri().startswith("sqlite"):
            emplacement = (ARTIFACTS_DIR / nom).resolve().as_uri()
        mlflow.create_experiment(nom, artifact_location=emplacement)
    mlflow.set_experiment(nom)


def _type_modele(modele):
    return "xgboost" if type(modele).__module__.startswith("xgboost") else "sklearn"


def log_run(experience, nom_run, params=None, metriques=None, modele=None,
            exemple_entree=None, artefacts=None, tags=None, nom_registry=None):
    """Enregistre un entraînement dans MLflow et renvoie son run_id.

    experience     : nom de l'expérience (ex. "regression", "alerte_precoce", "isolation_forest",
                     "cellule_zhang")
    nom_run        : nom lisible du run (ex. "random_forest_temp_max")
    params         : dict des hyperparamètres et choix (cible, features, seuil...)
    metriques      : dict nom -> nombre (MAE, F1, PR-AUC, rappel...)
    modele         : modèle entraîné (scikit-learn ou XGBoost), optionnel
    exemple_entree : petit DataFrame d'entrée, pour enregistrer la signature du modèle
    artefacts      : liste de chemins de fichiers à joindre (graphiques, CSV...)
    tags           : dict de tags (ex. {"membre": "Ivan Jr"})
    nom_registry   : si fourni, le modèle est aussi inscrit dans le Model Registry sous ce nom
    """
    configurer_mlflow()
    _experience(experience)
    with mlflow.start_run(run_name=nom_run) as run:
        if params:
            mlflow.log_params({k: _texte_court(v) for k, v in params.items()})
        if metriques:
            mlflow.log_metrics({k: float(v) for k, v in metriques.items()})
        if tags:
            mlflow.set_tags(tags)
        for chemin in artefacts or []:
            mlflow.log_artifact(str(Path(chemin)))
        if modele is not None:
            options = dict(name="modele", input_example=exemple_entree,
                           registered_model_name=nom_registry)
            if _type_modele(modele) == "xgboost":
                mlflow.xgboost.log_model(modele, **options)
            else:
                mlflow.sklearn.log_model(modele, skops_trusted_types=TYPES_SKOPS_CONFIANCE,
                                         **options)
        return run.info.run_id


def _texte_court(valeur, limite=500):
    """MLflow limite la taille des paramètres : on tronque les longues listes."""
    texte = str(valeur)
    return texte if len(texte) <= limite else texte[: limite - 3] + "..."


def charger_modele(nom_registry, version="latest"):
    """Recharge un modèle du Model Registry (ex. pour le dashboard ou main.py)."""
    configurer_mlflow()
    uri = f"models:/{nom_registry}/{version}"
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception:  # modèle enregistré avec la saveur xgboost
        return mlflow.xgboost.load_model(uri)
