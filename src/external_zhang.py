"""Lecture sécurisée des .pkl du dataset Zhang et export en parquet.

Seuls numpy et OrderedDict sont autorisés : tout autre objet dans le
pickle est refusé, pour ne jamais exécuter de code inconnu.
"""
import io
import pickle

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