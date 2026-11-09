"""Kern, Teil 3: Welche Aufträge gibt ein Spediteur in den Pool?

Modell: Jeder Spediteur mit Aufträgen gibt GENAU EINEN seiner Stopps ab. Die abgegebenen Stopps werden danach bestmöglich auf ALLE Fahrzeuge verteilt
(alle n^a Zuordnungen, Touren exakt); die übrigen Stopps bleiben bei ihrem Spediteur. Ein Auswahlkriterium wählt je Spediteur den abzugebenden Stopp aus
Informationen, die der Spediteur selbst hat (eigene Tour, Lage der Depots und Stopps der anderen); die bestmögliche Wahl probiert alle `per ** a` Kombinationen.

Einheiten: `pooled_cost` (Kosten bei gegebenen Abgaben), `all_offer_costs` (alle Kombinationen), `pick_*` (je Kriterium ein Auswahlindex je Spediteur)."""

import itertools

import numpy as np

import aal_constants as C
from aal_algorithm import Solved


def active_carriers(solved: Solved) -> list[int]:
    return [i for i in range(solved.n) if solved.inst.own[i]]


def pooled_cost(solved: Solved, offers: dict[int, int]) -> float:
    """Gesamtkosten, wenn Spediteur i den Stopp offers[i] (Index unter seinen Stopps) abgibt und alle abgegebenen Stopps bestmöglich auf alle Fahrzeuge verteilt werden."""
    inst = solved.inst
    n = inst.n
    pooled = [inst.stops_of(i)[idx] for i, idx in offers.items()]
    pooled_bits = np.array([1 << j for j in pooled], dtype=np.int64)
    rest = []
    for c in range(n):
        mask = inst.own[c]
        for j in pooled:
            if (mask >> j) & 1:
                mask ^= 1 << j
        rest.append(mask)
    a = len(pooled)
    assign = np.indices((n,) * a).reshape(a, -1).T            # (n^a, a): Fahrzeug je abgegebenem Stopp
    total = np.zeros(len(assign))
    for c in range(n):
        masks = rest[c] + ((assign == c) * pooled_bits[None, :]).sum(axis=1)
        total += solved.tours[c][masks]
    return float(total.min())


def all_offer_costs(solved: Solved) -> tuple[list[int], np.ndarray]:
    """Kosten aller Abgabe-Kombinationen: (aktive Spediteure, Feld der Form (per,)*a, Achse i = Wahl von Spediteur act[i])."""
    inst = solved.inst
    act = active_carriers(solved)
    sizes = [len(inst.stops_of(i)) for i in act]
    out = np.zeros(sizes)
    for combo in itertools.product(*[range(s) for s in sizes]):
        out[combo] = pooled_cost(solved, dict(zip(act, combo)))
    return act, out


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def pick_teuerster(solved: Solved, i: int) -> int:
    """Der Stopp, dessen Wegnahme die eigene Tour am meisten verkürzt (größte Entlastung); Gleichstand: kleinerer Index."""
    inst = solved.inst
    mine = inst.stops_of(i)
    full = inst.own[i]
    gain = [solved.tours[i][full] - solved.tours[i][full ^ (1 << j)] for j in mine]
    return int(np.argmax(gain))


def pick_fremdes_depot(solved: Solved, i: int) -> int:
    """Der Stopp mit der kleinsten Entfernung zu einem fremden Depot."""
    inst = solved.inst
    others = [k for k in range(inst.n) if k != i]
    dist = [min(_dist(inst.stops[j], inst.depots[k]) for k in others) for j in inst.stops_of(i)]
    return int(np.argmin(dist))


def pick_fremder_stopp(solved: Solved, i: int) -> int:
    """Der Stopp mit der kleinsten Entfernung zu einem Stopp eines anderen Spediteurs (gibt es keinen fremden Stopp, ersatzweise fremdes Depot)."""
    inst = solved.inst
    foreign = [j for j in range(inst.m) if inst.owner[j] != i]
    if not foreign:
        return pick_fremdes_depot(solved, i)
    dist = [min(_dist(inst.stops[j], inst.stops[f]) for f in foreign) for j in inst.stops_of(i)]
    return int(np.argmin(dist))


def pick_weit_vom_depot(solved: Solved, i: int) -> int:
    """Der Stopp mit der größten Entfernung zum eigenen Depot."""
    inst = solved.inst
    return int(np.argmax([_dist(inst.stops[j], inst.depots[i]) for j in inst.stops_of(i)]))


def pick_isoliert(solved: Solved, i: int) -> int:
    """Der Stopp mit der größten mittleren Entfernung zu den eigenen anderen Stopps (hat der Spediteur nur einen Stopp: Index 0)."""
    inst = solved.inst
    mine = inst.stops_of(i)
    if len(mine) == 1:
        return 0
    avg = [np.mean([_dist(inst.stops[j], inst.stops[k]) for k in mine if k != j]) for j in mine]
    return int(np.argmax(avg))


PICKERS = {
    "teuerster": pick_teuerster,
    "fremdes_depot": pick_fremdes_depot,
    "fremder_stopp": pick_fremder_stopp,
    "weit_vom_depot": pick_weit_vom_depot,
    "isoliert": pick_isoliert,
}


def criterion_savings(solved: Solved) -> dict[str, float]:
    """Ersparnis gegen "jeder fährt allein" (Anteil) je Kriterium. 'zufall' ist der Erwartungswert über alle Kombinationen (jede gleich wahrscheinlich)."""
    alone = float(solved.alone.sum())
    act, grid = all_offer_costs(solved)
    out = {"zufall": 1.0 - float(grid.mean()) / alone, "bestmoeglich": 1.0 - float(grid.min()) / alone}
    for name, picker in PICKERS.items():
        combo = tuple(picker(solved, i) for i in act)
        out[name] = 1.0 - float(grid[combo]) / alone
    return out


def criterion_picks(solved: Solved) -> dict[str, tuple[int, ...]]:
    """Welchen Stopp (Nummer unter den eigenen Stopps, ab 1) gibt jeder aktive Spediteur je Regel ab? Für die Beispielspur in der App."""
    act = active_carriers(solved)
    return {name: tuple(picker(solved, i) + 1 for i in act) for name, picker in PICKERS.items()}
