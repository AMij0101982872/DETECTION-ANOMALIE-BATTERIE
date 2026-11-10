import pandas as pd

from src.cleaning import nettoyer, compter_erreurs


def _ligne(vehicle_id=1, time="2019-07-26 20:00:00", current=10.0,
        cell_v_max=3.9, cell_v_min=3.8, temp_max=30, temp_min=28):
    return dict(vehicle_id=vehicle_id, time=pd.Timestamp(time), soc=50.0,
                pack_v=330.0, current=current, cell_v_max=cell_v_max,
                cell_v_min=cell_v_min, temp_max=temp_max, temp_min=temp_min,
                energy=10.0, capacity=30.0)


def _df(*lignes):
    return pd.DataFrame(list(lignes))


def test_courant_moins_400_est_retire():
    df = _df(
        _ligne(time="2019-07-26 20:00:00", current=10.0),
        _ligne(time="2019-07-26 20:00:08", current=-400.0),   # code d'erreur
        _ligne(time="2019-07-26 20:00:16", current=-399.0),   # valeur réelle, à garder
    )
    propre = nettoyer(df)
    assert len(propre) == 2
    assert (propre["current"] > -400).all()
    assert compter_erreurs(df)["courant = -400 A"] == 1


def test_tension_zero_est_retiree():
    df = _df(_ligne(), _ligne(time="2019-07-26 20:00:08", cell_v_min=0.0))
    assert len(nettoyer(df)) == 1


def test_doublon_est_retire():
    df = _df(_ligne(), _ligne())
    assert len(nettoyer(df)) == 1