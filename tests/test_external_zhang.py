import os
import pickle
from collections import OrderedDict

import numpy as np
import pytest

from src.external_zhang import safe_loads


class Malveillant:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_refuse_fonction_interdite():
    data = pickle.dumps(Malveillant())
    with pytest.raises(pickle.UnpicklingError):
        safe_loads(data)


def test_accepte_donnees_autorisees():
    data = pickle.dumps(OrderedDict(a=np.arange(3)))
    result = safe_loads(data)
    assert list(result["a"]) == [0, 1, 2]