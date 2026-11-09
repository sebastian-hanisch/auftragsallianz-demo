"""Kern gegen Handrechnung (Mini-Instanzen aus conftest.py) und Struktur der Hilfsfunktionen (DEMO-PLAYBOOK §1b).

Die unabhängigen Gegenproben (Permutationen, Vollaufzählung, Bondareva-Shapley, Gittersuche) stehen in test_oracle_auftragsallianz.py."""

import numpy as np
import pytest

import aal_algorithm as A
import aal_game as G
import aal_pool as P


def test_held_karp_on_the_3_4_5_triangle(triangle):
    depot, stops = triangle
    t = A.tour_costs(depot, stops)
    assert t[0] == 0.0
    assert t[0b01] == pytest.approx(6.0) and t[0b10] == pytest.approx(10.0) and t[0b11] == pytest.approx(12.0)


def test_tour_costs_of_no_stops_is_a_single_zero():
    assert A.tour_costs(np.array([0.0, 0.0]), np.zeros((0, 2))).tolist() == [0.0]


def test_held_karp_table_marks_impossible_endings_with_infinity(triangle):
    depot, stops = triangle
    dp, d0, _ = A.held_karp(depot, stops)
    assert dp[0b01, 1] == np.inf and dp[0b01, 0] == pytest.approx(3.0) and d0.tolist() == [3.0, 5.0]


def test_submask_pairs_have_3_to_the_m_entries_in_groups_by_mask():
    for m in (1, 3, 5):
        pm, ps, starts = A.submask_pairs(m)
        assert len(pm) == 3 ** m
        assert np.all((pm & ps) == ps)                                   # sub ist Teilmenge von mask
        assert np.all(np.diff(pm) >= 0)                                  # nach mask gruppiert
        assert starts[0] == 0 and starts[-1] == len(pm) - (1 << bin((1 << m) - 1).count("1"))


def test_min_plus_cover_on_one_stop():
    """m = 1: f_prev = [0, 5] (leer kostet 0, der Stopp beim Vorgänger 5), t = [0, 3]: Stopp beim neuen Fahrzeug kostet 3."""
    out = A.min_plus_cover(np.array([0.0, 5.0]), np.array([0.0, 3.0]), 1)
    assert out.tolist() == [0.0, 3.0]


def test_coalition_costs_of_the_swap_instance(swap):
    s = A.solve(swap)
    assert s.values.tolist() == pytest.approx([0.0, 18.0, 18.0, 4.0])
    assert s.grand == pytest.approx(4.0) and s.saving == pytest.approx(8 / 9)
    assert s.alone.tolist() == pytest.approx([18.0, 18.0])


def test_best_assignment_gives_each_vehicle_the_near_stop(swap):
    s = A.solve(swap)
    assert A.optimal_assignment(s) == [1, 0]                              # Stopp 0 (bei 9) fährt Fahrzeug 1, Stopp 1 (bei 1) fährt Fahrzeug 0
    assert A.optimal_assignment(s, (0,)) == [0, -1]                       # Spediteur 0 allein: nur seine eigenen Stopps
    assert A.check_feasible(swap, [1, 0]) and not A.check_feasible(swap, [1]) and not A.check_feasible(swap, [1, 2])


def test_tour_order_visits_the_triangle_stops_and_starts_near_the_depot(triangle):
    depot, stops = triangle
    assert sorted(A.tour_order(depot, stops, 0b11)) == [0, 1]
    assert A.tour_order(depot, stops, 0b10) == [1] and A.tour_order(depot, stops, 0) == []


def test_shapley_and_proportional_on_the_swap_instance(swap):
    s = A.solve(swap)
    assert G.shapley_costs(s.values, 2).tolist() == pytest.approx([2.0, 2.0])
    assert G.proportional_costs(s.values, 2).tolist() == pytest.approx([2.0, 2.0])
    assert G.least_core_epsilon(s.values, 2) == pytest.approx(-16.0)
    assert not G.core_is_empty(s.values, 2)
    assert G.in_core(G.shapley_costs(s.values, 2), s.values, 2)
    assert not G.in_core(np.array([20.0, -16.0]), s.values, 2)            # Spediteur 0 zahlt mehr als allein (18)


def test_max_excess_and_efficiency_in_in_core():
    values = np.array([0.0, 10.0, 10.0, 12.0])
    assert G.max_excess(np.array([6.0, 6.0]), values, 2) == pytest.approx(-4.0)
    assert not G.in_core(np.array([6.0, 5.0]), values, 2)                 # Summe 11 statt 12: nicht effizient


def test_an_empty_core_is_detected_on_a_three_player_cost_game():
    """Drei Spieler, jeder allein 10, jedes Paar 10, alle 18. Handrechnung: Die drei Paar-Bedingungen summiert geben 2 * 18 <= 30 + 3 * eps, also eps >= 2;
    x = (6, 6, 6) erreicht eps = 2 (Paare 12 <= 12, einzelne 6 <= 12). Der Kern ist leer."""
    values = np.zeros(8)
    for i in range(3):
        values[1 << i] = 10.0
    for pair in (3, 5, 6):
        values[pair] = 10.0
    values[7] = 18.0
    assert G.least_core_epsilon(values, 3) == pytest.approx(2.0)
    assert G.core_is_empty(values, 3)


def test_pool_cost_of_the_swap_instance(swap):
    s = A.solve(swap)
    assert P.pooled_cost(s, {0: 0, 1: 0}) == pytest.approx(4.0)           # beide geben ihren einzigen Stopp ab
    act, grid = P.all_offer_costs(s)
    assert act == [0, 1] and grid.shape == (1, 1) and grid[0, 0] == pytest.approx(4.0)
    assert P.criterion_savings(s)["bestmoeglich"] == pytest.approx(8 / 9)


def test_pick_rules_on_a_hand_instance():
    """Spediteur 0: Depot (0,0), Stopps a = (1,0) und b = (5,0); Spediteur 1: Depot (10,0), Stopp c = (6,0).
    teuerster: ohne b spart die Tour 10 - 2 = 8, ohne a nur 10 - 10 = 0 (b liegt hinter a auf der Linie): b.
    fremdes Depot: a ist 9 entfernt, b 5: b. fremder Stopp: a 5, b 1: b. weit vom eigenen Depot: b. isoliert: beide gleich weit (4) von den eigenen anderen: Gleichstand, kleinerer Index a."""
    from conftest import build_instance
    s = A.solve(build_instance([(0, 0), (10, 0)], [(1, 0), (5, 0), (6, 0)], [0, 0, 1]))
    assert P.pick_teuerster(s, 0) == 1 and P.pick_fremdes_depot(s, 0) == 1 and P.pick_fremder_stopp(s, 0) == 1
    assert P.pick_weit_vom_depot(s, 0) == 1 and P.pick_isoliert(s, 0) == 0
    assert P.pick_isoliert(s, 1) == 0                                    # nur ein Stopp: Index 0
    assert P.criterion_picks(s)["teuerster"] == (2, 1)
