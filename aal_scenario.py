"""Szenario: Spediteure mit eigenem Depot und Stopps im Einheitsquadrat.

Zufall nur über `aal_rng.SplitMix64` (Portfolio-Konvention). Je Stopp werden immer ein Paar Normalzahlen UND ein Paar Gleichverteilte gezogen,
und erst danach entscheidet die Streuung, welche gelten: Instanzen mit gleichem Zufalls-Seed und verschiedener Streuung sind gekoppelt
(gleiche Depots, gleiche Zufallszahlen), der Vergleich über die Streuung ist also ein Vergleich derselben Welt."""

import math
from dataclasses import dataclass, field

import numpy as np

import aal_constants as C
from aal_rng import SplitMix64


@dataclass(frozen=True, eq=False)
class Instance:
    depots: np.ndarray                 # (n, 2): ein Depot und ein Fahrzeug je Spediteur
    stops: np.ndarray                  # (M, 2): alle Stopps, nach Spediteuren geordnet
    owner: tuple[int, ...]             # owner[j] = Spediteur, dem Stopp j gehört
    own: tuple[int, ...] = field(default=())   # own[i] = Bitmaske der Stopps von Spediteur i (0 bei Spediteuren ohne Aufträge)

    @property
    def n(self) -> int:
        return len(self.depots)

    @property
    def m(self) -> int:
        return len(self.stops)

    def stops_of(self, i: int) -> list[int]:
        """Indizes der Stopps von Spediteur i, aufsteigend."""
        return [j for j, o in enumerate(self.owner) if o == i]


def _normal_pair(rng: SplitMix64) -> tuple[float, float]:
    """Zwei Standardnormalzahlen nach Box-Muller aus zwei Gleichverteilten."""
    u1 = 1.0 - rng.uniform()          # in (0, 1]
    u2 = rng.uniform()
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(2.0 * math.pi * u2), r * math.sin(2.0 * math.pi * u2)


def make_instance(n: int, per: int, spread_pct: int, idle: int, seed: int) -> Instance:
    """Deterministisch aus den Eingaben. Spediteure 0 bis n - idle - 1 haben je `per` Stopps, die übrigen nur Depot und Fahrzeug.

    spread_pct < SPREAD_MAX: Stopps normalverteilt um das eigene Depot mit Standardabweichung spread_pct / 100 (nicht auf das Gebiet beschnitten);
    spread_pct = SPREAD_MAX: Stopps gleichverteilt im Einheitsquadrat."""
    if not (1 <= n - idle and 0 <= idle and per >= 1):
        raise ValueError("mindestens ein Spediteur braucht Aufträge")
    rng = SplitMix64(seed)
    depots = np.array([[rng.uniform(), rng.uniform()] for _ in range(n)])
    active = n - idle
    pts, owner = [], []
    for i in range(active):
        for _ in range(per):
            zx, zy = _normal_pair(rng)
            ux, uy = rng.uniform(), rng.uniform()
            if spread_pct >= C.SPREAD_MAX:
                pts.append((ux, uy))
            else:
                s = spread_pct / 100.0
                pts.append((depots[i, 0] + s * zx, depots[i, 1] + s * zy))
            owner.append(i)
    own = tuple(sum(1 << j for j, o in enumerate(owner) if o == i) for i in range(n))
    return Instance(depots, np.array(pts), tuple(owner), own)
