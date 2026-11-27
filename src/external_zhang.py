"""Dataset Zhang et al. (2023) : lecture sécurisée des .pkl et export en parquet.

Seuls numpy et OrderedDict sont autorisés dans les pickles : tout autre objet
est refusé, pour ne jamais exécuter de code inconnu.

Indicateurs par extrait (voir data/external/LISEZMOI.md) :
    ecart de tension = (max_single_volt - min_single_volt) * 1000  [mV]
    ecart_moy_mV, ecart_max_mV : moyenne et maximum sur l'extrait
    temp_max = max(max_temp) ; dtemp = max(max_temp) - min(min_temp)
    courant_moy = moyenne du courant (négatif = charge)
    soc_debut = SOC de la première mesure
"""
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

TAILLE_EXTRAIT = 128  # mesures par extrait (une toutes les 10 s, ~21 min)

# Noms des champs dans les .pkl. À ADAPTER si ton fichier utilise d'autres clés
# (vérifie avec : print(list(safe_load("fichier.pkl").keys()))).
CLES = {
    "max_volt": "max_single_volt",
    "min_volt": "min_single_volt",
    "max_temp": "max_temp",
    "min_temp": "min_temp",
    "courant": "current",
    "soc": "soc",
}

COLONNES_SORTIE = [
    "marque", "split", "car", "label",
    "ecart_moy_mV", "ecart_max_mV", "temp_max", "dtemp",
    "courant_moy", "soc_debut",
]

# --------------------------------------------------------------------------
# Lecteur sécurisé
# --------------------------------------------------------------------------
ALLOWED = {
    ("collections", "OrderedDict"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Interdit : {module}.{name}")


def safe_loads(data: bytes):
    """Charge un pickle depuis des octets, en mode sécurisé."""
    return SafeUnpickler(io.BytesIO(data)).load()


def safe_load(path):
    """Charge un fichier .pkl, en mode sécurisé."""
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


# --------------------------------------------------------------------------
# Calcul des indicateurs
# --------------------------------------------------------------------------
def calculer_indicateurs(max_volt, min_volt, max_temp, min_temp, courant, soc):
    """Indicateurs d'un extrait (128 mesures), comme décrit dans LISEZMOI.md."""
    max_volt, min_volt = np.asarray(max_volt), np.asarray(min_volt)
    max_temp, min_temp = np.asarray(max_temp), np.asarray(min_temp)
    ecart_mV = (max_volt - min_volt) * 1000
    return {
        "ecart_moy_mV": float(ecart_mV.mean()),
        "ecart_max_mV": float(ecart_mV.max()),
        "temp_max": float(max_temp.max()),
        "dtemp": float(max_temp.max() - min_temp.min()),
        "courant_moy": float(np.mean(courant)),
        "soc_debut": float(np.asarray(soc)[0]),
    }


def decouper_en_extraits(mesures, taille=TAILLE_EXTRAIT):
    """Découpe les mesures d'un véhicule en extraits de `taille` mesures.

    `mesures` : dict de tableaux de même longueur, avec les clés de CLES.
    Le dernier morceau incomplet est ignoré. Retourne une liste de dicts
    d'indicateurs (un par extrait).
    """
    n = len(mesures[CLES["max_volt"]])
    resultats = []
    for debut in range(0, n - taille + 1, taille):
        tranche = slice(debut, debut + taille)
        resultats.append(
            calculer_indicateurs(
                max_volt=np.asarray(mesures[CLES["max_volt"]])[tranche],
                min_volt=np.asarray(mesures[CLES["min_volt"]])[tranche],
                max_temp=np.asarray(mesures[CLES["max_temp"]])[tranche],
                min_temp=np.asarray(mesures[CLES["min_temp"]])[tranche],
                courant=np.asarray(mesures[CLES["courant"]])[tranche],
                soc=np.asarray(mesures[CLES["soc"]])[tranche],
            )
        )
    return resultats


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def construire_table(vehicules, taille=TAILLE_EXTRAIT):
    """Construit la table d'extraits à partir d'une liste de véhicules.

    Chaque véhicule est un dict : {"marque", "split", "car", "label", "mesures"}
    où `mesures` est un dict de tableaux (voir decouper_en_extraits).
    """
    lignes = []
    for v in vehicules:
        for indic in decouper_en_extraits(v["mesures"], taille):
            lignes.append(
                {
                    "marque": v["marque"],
                    "split": v["split"],
                    "car": v["car"],
                    "label": int(v["label"]),
                    **indic,
                }
            )
    return pd.DataFrame(lignes, columns=COLONNES_SORTIE)


def exporter_parquet(vehicules, chemin_sortie, taille=TAILLE_EXTRAIT):
    """Écrit la table d'extraits au format parquet et retourne le DataFrame."""
    table = construire_table(vehicules, taille)
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(chemin_sortie, index=False)
    return table