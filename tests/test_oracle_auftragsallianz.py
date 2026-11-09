"""Orakel-Tests auftragsallianz-demo: fünf Kernroutinen gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2 und §5).

1. Held-Karp gegen Aufzählung aller Permutationen (bis 7 Stopps).
2. Verteilungs-DP (Koalitionskosten) gegen Vollaufzählung aller Zuordnungen (n hoch k), Touren per Permutationen.
3. Kern-LP (Least Core) gegen Bondareva-Shapley (duales LP) und gegen eine Gittersuche bei n = 3.
4. Shapley-Wert gegen den Mittelwert über alle Permutationen der Eintrittsreihenfolge.
5. Bestmögliche Pool-Wahl gegen Vollaufzählung aus Wahl und Zuordnung.

Die Handbeispiele zu den Orakeln selbst stehen am Ende (test_oracle_*_on_a_hand_example). Der große Lauf (mindestens 200 Instanzen je Orakel) wurde im Scratch-Ordner gemacht
und steht in der README unter Verifikation; hier läuft die kleine, schnelle Fassung."""

import itertools
import math

import numpy as np
import pytest

import aal_algorithm as A
import aal_game as G
import aal_pool as P
from aal_scenario import make_instance
from conftest import build_instance


def perm_tour(depot, pts):
    """Orakel-Tour: kleinste Länge über alle Permutationen der Punkte (Depot, Punkte, Depot)."""
    if len(pts) == 0:
        return 0.0
    best = math.inf
    for order in itertools.permutations(range(len(pts))):
        length, cur = 0.0, depot
        for k in order:
            length += float(np.hypot(*(pts[k] - cur)))
            cur = pts[k]
        length += float(np.hypot(*(depot - cur)))
        best = min(best, length)
    return best


def perm_assignment_cost(inst, members, stops_of_members):
    """Orakel: kleinste Gesamtkosten über alle Zuordnungen der Stopps (Index-Liste) zu den Fahrzeugen der Mitglieder, Tour je Fahrzeug per Permutationen."""
    best = math.inf
    for assign in itertools.product(range(len(members)), repeat=len(stops_of_members)):
        total = 0.0
        for v, c in enumerate(members):
            pts = [inst.stops[j] for j, a in zip(stops_of_members, assign) if a == v]
            total += perm_tour(inst.depots[c], np.array(pts).reshape(-1, 2))
        best = min(best, total)
    return best


def test_held_karp_against_all_permutations():
    for seed in range(12):
        inst = make_instance(2, 3, 100, 0, seed)                      # 6 Stopps
        stops = inst.stops
        t = A.tour_costs(inst.depots[0], stops)
        for mask in range(1 << len(stops)):
            pts = np.array([stops[j] for j in range(len(stops)) if (mask >> j) & 1]).reshape(-1, 2)
            assert t[mask] == pytest.approx(perm_tour(inst.depots[0], pts), rel=1e-9), (seed, mask)
    big = make_instance(1 + 1, 3, 100, 0, 99).stops                    # 7 Stopps: nur die volle Maske
    inst7 = build_instance([(0.5, 0.5), (0.1, 0.9)], np.vstack([big, [[0.3, 0.2]]]), [0, 0, 0, 1, 1, 1, 1])
    assert A.tour_costs(inst7.depots[0], inst7.stops)[-1] == pytest.approx(perm_tour(inst7.depots[0], inst7.stops), rel=1e-9)


def test_coalition_costs_against_full_enumeration_of_assignments():
    for seed in range(10):
        inst = make_instance(3, 2, 100, 0, seed)                      # 3 Spediteure, 6 Stopps: 3^6 = 729 Zuordnungen für die große Koalition
        s = A.solve(inst)
        for coalition in range(1, 8):
            members = [i for i in range(3) if (coalition >> i) & 1]
            stops = [j for j in range(inst.m) if inst.owner[j] in members]
            assert s.values[coalition] == pytest.approx(perm_assignment_cost(inst, members, stops), rel=1e-9), (seed, coalition)


def test_coalition_costs_with_idle_carriers_against_enumeration():
    for seed in range(6):
        inst = make_instance(3, 3, 40, 1, seed)                       # 2 Spediteure mit je 3 Stopps, einer ohne
        s = A.solve(inst)
        assert s.values[1 << 2] == 0.0                                  # der Spediteur ohne Aufträge kostet allein nichts
        stops = list(range(inst.m))
        assert s.grand == pytest.approx(perm_assignment_cost(inst, [0, 1, 2], stops), rel=1e-9)


def bondareva_shapley_value(values, n):
    """Orakel: Dual des Kern-LP. min sum_S lambda_S C(S) mit sum_{S enthält i} lambda_S = 1 (alle nicht leeren S). Kern nicht leer genau dann, wenn Wert >= C(N)."""
    from scipy.optimize import linprog
    sets = list(range(1, 1 << n))
    a_eq = np.array([[1.0 if (s >> i) & 1 else 0.0 for s in sets] for i in range(n)])
    res = linprog([values[s] for s in sets], A_eq=a_eq, b_eq=np.ones(n), bounds=[(0, None)] * len(sets), method="highs")
    assert res.status == 0
    return float(res.fun)


def grid_min_excess(values, grid=60):
    """Orakel n = 3: kleinster größter Überschuss über ein Gitter effizienter Aufteilungen (x0, x1 auf dem Gitter, x2 ergibt sich)."""
    grand = values[7]
    lo, hi = -0.5 * grand, 1.5 * grand
    best = math.inf
    for x0 in np.linspace(lo, hi, grid):
        for x1 in np.linspace(lo, hi, grid):
            x = np.array([x0, x1, grand - x0 - x1])
            best = min(best, G.max_excess(x, values, 3))
    return best


def test_least_core_against_bondareva_shapley_dual_and_grid_search():
    pytest.importorskip("scipy")
    for seed in range(12):
        s = A.solve(make_instance(3, 2, 30, 0, seed))
        eps = G.least_core_epsilon(s.values, 3)
        dual = bondareva_shapley_value(s.values, 3)
        assert (dual >= s.grand - 1e-9) == (eps <= 1e-9), (seed, eps, dual)      # Kern nicht leer: beide Wege sagen dasselbe
        grid = grid_min_excess(s.values)
        assert grid >= eps - 1e-9                                                  # kein Gitterpunkt unterbietet das LP
        assert grid <= eps + 0.1 * max(1.0, s.grand)                               # und das Gitter kommt dem LP nahe (Gitterweite begrenzt die Genauigkeit)


def test_bondareva_shapley_dual_equals_core_lp_value_on_random_games():
    """Stärker als das Vorzeichen: max sum x s.t. x(S) <= C(S) hat denselben Wert wie das Dual; ist der Kern nicht leer, ist es C(N), sonst kleiner."""
    rng = np.random.default_rng(5)
    for _ in range(25):
        n = 4
        values = np.zeros(1 << n)
        for s in range(1, 1 << n):
            values[s] = rng.uniform(1, 5) * bin(s).count("1") ** 0.8
        eps = G.least_core_epsilon(values, n)
        dual = bondareva_shapley_value(values, n)
        assert (dual >= values[-1] - 1e-9) == (eps <= 1e-9), (eps, dual)


def test_shapley_against_the_mean_over_all_permutations():
    for seed in range(8):
        s = A.solve(make_instance(4, 2, 100, 0, seed))
        n = 4
        acc = np.zeros(n)
        perms = list(itertools.permutations(range(n)))
        for perm in perms:
            built = 0
            for i in perm:
                acc[i] += s.values[built | (1 << i)] - s.values[built]
                built |= 1 << i
        assert G.shapley_costs(s.values, n) == pytest.approx(acc / len(perms), rel=1e-9, abs=1e-12)
        assert G.shapley_costs(s.values, n).sum() == pytest.approx(s.grand, rel=1e-9)       # Effizienz


def test_best_pool_choice_against_full_enumeration():
    for seed in range(6):
        inst = make_instance(3, 2, 100, 0, seed)                      # 3 Spediteure, 2 Stopps: 8 Wahlen, je 27 Zuordnungen
        s = A.solve(inst)
        act, grid = P.all_offer_costs(s)
        best = math.inf
        for offers in itertools.product(range(2), repeat=3):
            pooled = [inst.stops_of(i)[o] for i, o in enumerate(offers)]
            rest = [[j for j in inst.stops_of(c) if j not in pooled] for c in range(3)]
            for assign in itertools.product(range(3), repeat=3):
                total = 0.0
                for c in range(3):
                    idx = rest[c] + [pooled[k] for k, v in enumerate(assign) if v == c]
                    total += perm_tour(inst.depots[c], np.array([inst.stops[j] for j in idx]).reshape(-1, 2))
                best = min(best, total)
        assert float(grid.min()) == pytest.approx(best, rel=1e-9), seed


def test_oracles_on_a_hand_example(swap):
    """Die Orakel selbst an der Handrechnung: Tausch-Instanz, C = [0, 18, 18, 4], Dual-Wert 4 (Kern nicht leer), Shapley (2, 2)."""
    assert perm_tour(swap.depots[0], swap.stops[[0]]) == pytest.approx(18.0)
    assert perm_assignment_cost(swap, [0, 1], [0, 1]) == pytest.approx(4.0)
    assert bondareva_shapley_value(np.array([0.0, 18.0, 18.0, 4.0]), 2) == pytest.approx(4.0)
    assert grid_min_excess_two_players(np.array([0.0, 18.0, 18.0, 4.0])) <= -15.9
    # Spiel mit leerem Kern (jeder allein 10, jedes Paar 10, alle 18; Handrechnung Least-Core-Wert 2): Dual-Wert 15 (je Paar 1/2) liegt unter C(N) = 18
    empty = np.zeros(8)
    empty[[1, 2, 4]] = 10.0
    empty[[3, 5, 6]] = 10.0
    empty[7] = 18.0
    assert bondareva_shapley_value(empty, 3) == pytest.approx(15.0)
    assert G.least_core_epsilon(empty, 3) == pytest.approx(2.0) and grid_min_excess(empty) >= 2.0 - 1e-9


def grid_min_excess_two_players(values):
    """Gitter-Orakel für n = 2 (nur für das Handbeispiel): kleinster größter Überschuss über x0 in [-10, 40]."""
    best = math.inf
    for x0 in np.linspace(-10, 40, 501):
        best = min(best, G.max_excess(np.array([x0, values[3] - x0]), values, 2))
    return best
