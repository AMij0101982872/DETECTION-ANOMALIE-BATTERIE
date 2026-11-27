import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from src.external_zhang import (
    CLES,
    COLONNES_SORTIE,
    calculer_indicateurs,
    decouper_en_extraits,
    exporter_parquet,
    safe_loads,
)


class Malveillant:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _mesures(n, ecart_v=0.02):
    """Mesures factices de n points."""
    return {
        CLES["max_volt"]: np.full(n, 3.60 + ecart_v),
        CLES["min_volt"]: np.full(n, 3.60),
        CLES["max_temp"]: np.full(n, 30.0),
        CLES["min_temp"]: np.full(n, 28.0),
        CLES["courant"]: np.full(n, -10.0),
        CLES["soc"]: np.linspace(50, 60, n),
    }


# --- lecteur sécurisé ---
def test_refuse_fonction_interdite():
    data = pickle.dumps(Malveillant())
    with pytest.raises(pickle.UnpicklingError):
        safe_loads(data)


def test_accepte_donnees_autorisees():
    data = pickle.dumps(OrderedDict(a=np.arange(3)))
    result = safe_loads(data)
    assert list(result["a"]) == [0, 1, 2]


# --- indicateurs ---
def test_calculer_indicateurs():
    r = calculer_indicateurs(
        max_volt=[3.60, 3.62], min_volt=[3.58, 3.57],
        max_temp=[30, 32], min_temp=[28, 29],
        courant=[-10, -20], soc=[50, 55],
    )
    assert r["ecart_moy_mV"] == pytest.approx(35.0)
    assert r["ecart_max_mV"] == pytest.approx(50.0)
    assert r["temp_max"] == 32
    assert r["dtemp"] == 4
    assert r["courant_moy"] == -15
    assert r["soc_debut"] == 50


# --- découpage ---
def test_decoupage_en_extraits_complets():
    assert len(decouper_en_extraits(_mesures(256))) == 2


def test_decoupage_ignore_morceau_incomplet():
    assert len(decouper_en_extraits(_mesures(300))) == 2


def test_decoupage_trop_court():
    assert decouper_en_extraits(_mesures(100)) == []


# --- export ---
def test_export_parquet(tmp_path):
    vehicules = [
        {"marque": 1, "split": "train", "car": 7, "label": 1, "mesures": _mesures(256)},
        {"marque": 2, "split": "test", "car": 8, "label": 0, "mesures": _mesures(128)},
    ]
    sortie = tmp_path / "zhang_test.parquet"
    exporter_parquet(vehicules, sortie)

    df = pd.read_parquet(sortie)
    assert list(df.columns) == COLONNES_SORTIE
    assert len(df) == 3  # 2 extraits + 1 extrait
    assert df.loc[df["car"] == 7, "label"].eq(1).all()
    assert df["ecart_moy_mV"].iloc[0] == pytest.approx(20.0)