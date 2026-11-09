"""Pipeline complet en une commande (rapport §4.2) : python main.py

Étapes : fusion -> nettoyage -> sessions/features -> modèles -> fusion par vote.
Les étapes non encore codées lèvent NotImplementedError.
"""
from src import config
from src.cleaning import nettoyer
from src.data_loading import fusionner_csv
from src.sessions import ajouter_session_id, features_par_session


def main():
    print("Étape 2 : fusion des CSV")
    df = fusionner_csv(verbose=False)
    df.to_parquet(config.DATASET_PARQUET, index=False)

    print("Étape 3 : nettoyage")
    df = nettoyer(df)
    df.to_parquet(config.CLEAN_PARQUET, index=False)

    print("Étape 5 : sessions et features")
    sessions = features_par_session(ajouter_session_id(df))
    sessions.to_parquet(config.SESSIONS_PARQUET, index=False)

    print("Étape 6 : modèles + fusion par vote")
    # À brancher quand les modèles seront prêts :
    # sessions["vote_seuils"] = seuils.vote_sessions(sessions)
    # sessions["vote_regression"] = regression.vote_sessions(sessions, modele_reg)
    # sessions["vote_alerte_precoce"] = alerte_precoce.vote_sessions(sessions, modele_ap)
    # sessions["vote_isolation_forest"] = isolation_forest.vote_sessions(sessions, modele_if)
    # fusion.fusionner(sessions).to_parquet(config.RESULTATS_PARQUET, index=False)


if __name__ == "__main__":
    main()
