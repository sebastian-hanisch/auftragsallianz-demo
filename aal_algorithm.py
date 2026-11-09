"""Kern, Teil 1 (numpy-only): Tourkosten, Verteilung der Stopps auf Fahrzeuge, Koalitionskosten.

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
`held_karp` / `tour_costs` (kürzeste Rundtour ab Depot je Stopp-Teilmenge, exakt), `submask_pairs` + `min_plus_cover` (beste Aufteilung einer Stopp-Menge auf zwei
Fahrzeuggruppen), `coalition_costs` (Kosten C(S) jeder Koalition), `optimal_assignment` / `tour_order` (Rückverfolgung für die Karte), `solve` ruft nur der Reihe nach auf.
Stopp-Mengen sind Bitmasken (Bit j = Stopp j), Koalitionen Bitmasken über die Spediteure."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from aal_scenario import Instance


def popcount_table(m: int) -> np.ndarray:
    """Zahl der gesetzten Bits für alle Bitmasken 0 bis 2^m - 1."""
    pop = np.zeros(1 << m, dtype=np.int64)
    for b in range(m):
        pop += (np.arange(1 << m) >> b) & 1
    return pop


def held_karp(depot: np.ndarray, stops: np.ndarray):
    """Held-Karp-Tabelle: dp[mask, j] = kürzester Weg vom Depot über genau die Stopps der Maske, der in Stopp j endet (inf, wenn j nicht in der Maske).

    Gibt (dp, d0, d) zurück: d0 = Depot-Stopp-Entfernungen, d = Stopp-Stopp-Entfernungen (euklidisch)."""
    m = len(stops)
    d0 = np.linalg.norm(stops - depot, axis=1)
    d = np.linalg.norm(stops[:, None, :] - stops[None, :, :], axis=2)
    dp = np.full((1 << m, m), np.inf)
    for j in range(m):
        dp[1 << j, j] = d0[j]
    masks = np.arange(1 << m)
    pop = popcount_table(m)
    for p in range(2, m + 1):
        layer = masks[pop == p]
        for k in range(m):
            sel = layer[((layer >> k) & 1) == 1]
            prev = sel ^ (1 << k)
            dp[sel, k] = (dp[prev] + d[:, k][None, :]).min(axis=1)
    return dp, d0, d


def tour_costs(depot: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Länge der kürzesten Rundtour Depot, alle Stopps der Maske, Depot, für jede Maske (Länge 2^M); die leere Menge kostet 0."""
    m = len(stops)
    if m == 0:
        return np.zeros(1)
    dp, d0, _ = held_karp(depot, stops)
    out = (dp + d0[None, :]).min(axis=1)
    out[0] = 0.0
    return out


@lru_cache(maxsize=16)
def submask_pairs(m: int):
    """Alle Paare (Maske, Teilmenge davon) für m Stopps, nach Maske gruppiert: (mask, sub, starts); die Gruppe von Maske k beginnt bei starts[k].

    Die Gruppen haben 2^popcount(Maske) Einträge, insgesamt 3^m."""
    pm = np.zeros(1, dtype=np.int64)
    ps = np.zeros(1, dtype=np.int64)
    for b in range(m):
        bit = 1 << b
        pm = np.concatenate([pm, pm | bit, pm | bit])
        ps = np.concatenate([ps, ps | bit, ps])
    order = np.argsort(pm, kind="stable")
    pm, ps = pm[order], ps[order]
    counts = 1 << popcount_table(m)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return pm, ps, starts


def min_plus_cover(f_prev: np.ndarray, t: np.ndarray, m: int) -> np.ndarray:
    """Beste Aufteilung: out[mask] = min über Teilmengen sub von mask von f_prev[mask ohne sub] + t[sub] (ein Fahrzeug mehr kommt dazu)."""
    pm, ps, starts = submask_pairs(m)
    return np.minimum.reduceat(f_prev[pm ^ ps] + t[ps], starts)


@dataclass(frozen=True, eq=False)
class Solved:
    """Alles, was zu einer Instanz exakt berechnet ist (die Grundlage für Ersparnis, Aufteilung, Pool)."""
    inst: Instance
    tours: tuple[np.ndarray, ...]      # tours[i][mask] = Tourkosten des Fahrzeugs von Spediteur i über die Stopp-Menge mask
    values: np.ndarray                 # values[S] = C(S): beste Gesamtkosten der Koalition S (Bitmaske über Spediteure), values[0] = 0

    @property
    def n(self) -> int:
        return self.inst.n

    @property
    def grand(self) -> float:
        return float(self.values[(1 << self.n) - 1])

    @property
    def alone(self) -> np.ndarray:
        """Alleinkosten C({i}) je Spediteur."""
        return np.array([self.values[1 << i] for i in range(self.n)])

    @property
    def saving(self) -> float:
        """Ersparnis = 1 - C(alle) / Summe der Alleinkosten, als Anteil (0 bis 1). C(alle) <= Summe der Alleinkosten gilt mathematisch immer; ein winziger negativer Wert wäre Rundungsrauschen und wird auf 0 gesetzt."""
        total = float(self.alone.sum())
        return 0.0 if total <= 0 else max(0.0, 1.0 - self.grand / total)


def coalition_costs(tours, own: tuple[int, ...], m: int) -> np.ndarray:
    """Kosten C(S) jeder Koalition S: die Stopps aller Mitglieder werden bestmöglich auf die Fahrzeuge der Mitglieder verteilt.

    Dynamische Programmierung über die Koalitionen: f_S = f_(S ohne ein Mitglied c) verknüpft mit den Tourkosten von c."""
    n = len(tours)
    f: dict[int, np.ndarray] = {}
    values = np.zeros(1 << n)
    for s in range(1, 1 << n):
        low = s & -s
        c = low.bit_length() - 1
        rest = s ^ low
        f[s] = tours[c] if rest == 0 else min_plus_cover(f[rest], tours[c], m)
        union = 0
        for i in range(n):
            if (s >> i) & 1:
                union |= own[i]
        values[s] = f[s][union]
    return values


def solve(inst: Instance) -> Solved:
    tours = tuple(tour_costs(inst.depots[i], inst.stops) for i in range(inst.n))
    return Solved(inst, tours, coalition_costs(tours, inst.own, inst.m))


def check_feasible(inst: Instance, assignment: list[int]) -> bool:
    """Unabhängiger Prüfer: assignment[j] = Fahrzeug von Stopp j; jeder Stopp genau einem existierenden Fahrzeug."""
    return len(assignment) == inst.m and all(0 <= c < inst.n for c in assignment)


def optimal_assignment(solved: Solved, members: tuple[int, ...] | None = None) -> list[int]:
    """Beste Zuordnung der Stopps der Koalition `members` (Standard: alle) zu den Fahrzeugen der Mitglieder: Rückverfolgung der Verteilungs-DP.

    Gibt für JEDEN Stopp der Instanz das Fahrzeug zurück; Stopps von Nichtmitgliedern bekommen -1."""
    inst = solved.inst
    members = tuple(range(inst.n)) if members is None else tuple(members)
    union = 0
    for i in members:
        union |= inst.own[i]
    pm, ps, starts = submask_pairs(inst.m)
    prefix = [solved.tours[members[0]]]
    for c in members[1:]:
        prefix.append(min_plus_cover(prefix[-1], solved.tours[c], inst.m))
    assign = [-1] * inst.m
    mask = union
    for idx in range(len(members) - 1, 0, -1):
        lo = starts[mask]
        hi = lo + (1 << bin(mask).count("1"))
        subs = ps[lo:hi]
        val = prefix[idx - 1][mask ^ subs] + solved.tours[members[idx]][subs]
        best = int(subs[int(np.argmin(val))])
        for j in range(inst.m):
            if (best >> j) & 1:
                assign[j] = members[idx]
        mask ^= best
    for j in range(inst.m):
        if (mask >> j) & 1:
            assign[j] = members[0]
    return assign


def tour_order(depot: np.ndarray, stops: np.ndarray, mask: int) -> list[int]:
    """Reihenfolge der Stopps der Maske auf der kürzesten Rundtour (Rückverfolgung der Held-Karp-Tabelle); leer bei leerer Maske."""
    if mask == 0:
        return []
    dp, d0, d = held_karp(depot, stops)
    m = len(stops)
    j = int(np.argmin([dp[mask, k] + d0[k] if (mask >> k) & 1 else np.inf for k in range(m)]))
    order = [j]
    while mask & (mask - 1):
        prev = mask ^ (1 << j)
        j2 = int(np.argmin([dp[prev, k] + d[k, j] if (prev >> k) & 1 else np.inf for k in range(m)]))
        order.append(j2)
        mask, j = prev, j2
    return order[::-1]
