"""Approche 0 — Seuils (sans apprentissage) : « ça dépasse maintenant ».

Sert aussi à fabriquer les étiquettes du modèle 2 (rapport §3.4).
"""
import pandas as pd

from src.config import (ECART_TENSION_MAX, FENETRE_CONFIRMATION,
                        TEMP_ALERTE, TEMP_CRITIQUE)


def _confirme(df, col):
    """Anomalie confirmée = N mesures consécutives (écarte les pics de bruit)."""
    n = FENETRE_CONFIRMATION
    return (df.groupby("vehicle_id")[col]
              .transform(lambda s: s.astype(int).rolling(n).sum() == n))


def detecter_mesures(df):
    """Indicateurs au niveau de chaque mesure (8 s)."""
    df = df.copy()
    df["ecart_tension"] = df["cell_v_max"] - df["cell_v_min"]
    df["surchauffe_alerte"] = df["temp_max"] > TEMP_ALERTE
    df["surchauffe_critique"] = df["temp_max"] >= TEMP_CRITIQUE
    df["cellule_defaillante"] = df["ecart_tension"] > ECART_TENSION_MAX
    df["surchauffe_confirmee"] = _confirme(df, "surchauffe_alerte")
    df["cellule_confirmee"] = _confirme(df, "cellule_defaillante")
    return df


def resume_par_vehicule(df):
    return df.groupby("vehicle_id").agg(
        mesures=("time", "size"),
        temp_max=("temp_max", "max"),
        surchauffe_45=("surchauffe_alerte", "sum"),
        surchauffe_48=("surchauffe_critique", "sum"),
        surchauffe_conf=("surchauffe_confirmee", "sum"),
        ecart_v_max=("ecart_tension", "max"),
        ecart_v_moyen=("ecart_tension", "mean"),
        cellule_0_1V=("cellule_defaillante", "sum"),
        cellule_conf=("cellule_confirmee", "sum"),
    )


def vote_sessions(sessions):
    """Vote 0/1 par session. À FAIRE quand sessions.py sera prêt :
    1 si temp_max de la session > TEMP_ALERTE ou écart de tension max > ECART_TENSION_MAX."""
    raise NotImplementedError


if __name__ == "__main__":
    from src.cleaning import charger_nettoye
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(resume_par_vehicule(detecter_mesures(charger_nettoye())).round(3))
