"""Test de log_run : un run est bien enregistré (MLflow dans un dossier temporaire)."""
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import mlops


def test_log_run_enregistre_params_metriques_modele(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(mlops, "ARTIFACTS_DIR", tmp_path / "mlartifacts")
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    # modèle à base d'arbres : vérifie aussi la sauvegarde skops (types de confiance)
    modele = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, [0.0, 2.0, 4.0, 6.0])

    run_id = mlops.log_run("test", "foret", params={"cible": "y"},
                           metriques={"mae": 0.0}, modele=modele, exemple_entree=X,
                           nom_registry="test_foret")

    run = mlflow.get_run(run_id)
    assert run.data.params["cible"] == "y"
    assert run.data.metrics["mae"] == 0.0
    recharge = mlops.charger_modele("test_foret")
    assert recharge.predict(X).tolist() == modele.predict(X).tolist()
