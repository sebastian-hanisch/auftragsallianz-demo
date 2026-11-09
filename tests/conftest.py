"""Gemeinsame Fixtures: Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aal_scenario import Instance  # noqa: E402


def build_instance(depots, stops, owner) -> Instance:
    """Instanz aus Hand-Koordinaten (owner[j] = Spediteur von Stopp j)."""
    n = len(depots)
    own = tuple(sum(1 << j for j, o in enumerate(owner) if o == i) for i in range(n))
    return Instance(np.array(depots, dtype=float), np.array(stops, dtype=float), tuple(owner), own)


@pytest.fixture
def triangle():
    """Depot (0, 0), Stopps (3, 0) und (3, 4). Handrechnung: allein (3,0): 6; allein (3,4): 10; beide: 3 + 4 + 5 = 12 (rechtwinkliges 3-4-5-Dreieck)."""
    return np.array([0.0, 0.0]), np.array([[3.0, 0.0], [3.0, 4.0]])


@pytest.fixture
def swap():
    """Zwei Spediteure, die ihre Stopps tauschen sollten. Depots (0,0) und (10,0); Spediteur 0 hat den Stopp (9,0), Spediteur 1 den Stopp (1,0).

    Handrechnung: allein je 18 (Hin- und Rückweg 9 mal 2), gemeinsam 2 + 2 = 4 (jedes Fahrzeug fährt den nahen Stopp des anderen).
    Also C = [0, 18, 18, 4]; Ersparnis 1 - 4/36 = 8/9; Shapley und proportional je (2, 2); Least-Core-Wert (4 - 36)/2 = -16."""
    return build_instance([(0, 0), (10, 0)], [(9, 0), (1, 0)], [0, 1])
