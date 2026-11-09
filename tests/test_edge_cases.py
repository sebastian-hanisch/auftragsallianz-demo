"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 2, ein einziger Stopp, Spediteure ohne Aufträge, Gleichstände, Reglerränder,
Worst-Case am Regler-Maximum und Reproduzierbarkeit über Prozesse (Hash-Randomisierung, DEMO-PLAYBOOK §1d)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aal_algorithm as A
import aal_constants as C
import aal_evaluation as E
import aal_game as G
import aal_pool as P
from aal_scenario import make_instance
from conftest import build_instance

ROOT = Path(__file__).resolve().parent.parent


def test_exactly_two_carriers_and_one_stop_each():
    """n = 2, ein Stopp je Spediteur: der kleinste Fall mit etwas zu vergleichen (Kern-LP hat genau zwei Teilkoalitionen)."""
    s = A.solve(make_instance(2, 1, 100, 0, 3))
    assert s.values[3] <= s.alone.sum() + 1e-12
    assert G.least_core_epsilon(s.values, 2) == pytest.approx((s.grand - s.alone.sum()) / 2, abs=1e-9)    # für n = 2 gilt eps = (C(N) - C(1) - C(2)) / 2
    assert G.shapley_costs(s.values, 2).sum() == pytest.approx(s.grand)


def test_single_stop_and_single_carrier_with_orders():
    """Ein Spediteur mit Aufträgen, die anderen nur mit Fahrzeug: C(allein) > 0, Alleinkosten der Übrigen 0, kein Teilen durch 0 in Ersparnis und Anteilen."""
    inst = make_instance(3, 1, 30, 2, 4)
    s = A.solve(inst)
    assert inst.m == 1 and s.alone[1] == 0.0 and s.alone[2] == 0.0 and s.alone[0] > 0
    assert 0.0 <= s.saving <= 1.0
    assert G.proportional_costs(s.values, 3)[1:].tolist() == [0.0, 0.0]
    assert G.shapley_costs(s.values, 3).sum() == pytest.approx(s.grand)


def test_an_idle_carrier_never_pays_more_than_nothing():
    """Ein Spediteur ohne Aufträge kann der Koalition nur helfen: sein Shapley-Kostenanteil ist <= 0 (Grenzkosten eines zusätzlichen Fahrzeugs sind nie positiv)."""
    for seed in range(15):
        s = A.solve(make_instance(4, 3, 30, 2, seed))
        assert (G.shapley_costs(s.values, 4)[2:] <= 1e-12).all(), seed


def test_two_stops_at_the_same_place_and_a_stop_on_the_depot():
    """Gleichstand und Entartung: Stopps auf demselben Punkt kosten keinen Umweg, ein Stopp auf dem Depot kostet nichts."""
    inst = build_instance([(0, 0), (4, 0)], [(2, 0), (2, 0), (0, 0)], [0, 0, 1])
    s = A.solve(inst)
    assert s.alone[0] == pytest.approx(4.0)                              # beide Stopps bei 2: hin und zurück 4
    assert s.alone[1] == pytest.approx(8.0)                              # der Stopp auf Depot 0 liegt für Spediteur 1 vier Einheiten weg
    assert s.grand == pytest.approx(4.0)                                 # Fahrzeug 0 fährt alle drei, Fahrzeug 1 bleibt stehen
    assert A.check_feasible(inst, A.optimal_assignment(s))


def test_all_stops_identical_gives_ties_without_errors():
    inst = build_instance([(0, 0), (0, 0)], [(1, 1), (1, 1)], [0, 1])
    s = A.solve(inst)
    assert s.saving == pytest.approx(0.5)                                # gemeinsam fährt ein Fahrzeug beide Stopps, das andere bleibt stehen
    assert G.shapley_costs(s.values, 2).tolist() == pytest.approx([s.grand / 2] * 2)


def test_the_pool_with_one_stop_each_has_nothing_to_choose():
    s = A.solve(make_instance(3, 1, 100, 0, 5))
    sav = P.criterion_savings(s)
    assert len({round(v, 12) for v in sav.values()}) == 1       # alle Regeln, Zufall und bestmöglich gleich


def test_pool_savings_are_ordered_between_random_expectation_and_the_best_choice():
    for seed in range(10):
        sav = P.criterion_savings(A.solve(make_instance(4, 3, 30, 0, seed)))
        assert sav["bestmoeglich"] >= max(sav[c] for c in C.CRITERIA_RULES) - 1e-12
        assert sav["bestmoeglich"] >= sav["zufall"] - 1e-12


@pytest.mark.parametrize("n,idle", [(2, 0), (2, 1), (5, 0), (5, 4), (3, 2)])
def test_effective_settings_keep_every_combination_inside_the_exact_limit(n, idle):
    for per in (C.PER_MIN, C.PER_MAX):
        eff = E.effective_settings(E.Settings(n, per, 15, idle, 10, 1))
        assert 1 <= eff.n - eff.idle and eff.per >= 1
        assert eff.per * (eff.n - eff.idle) <= C.MAX_STOPS


def test_spread_extremes_run_and_the_largest_means_uniform():
    for spread in (C.SPREAD_MIN, C.SPREAD_MAX):
        s = A.solve(make_instance(4, 2, spread, 0, 2))
        assert 0.0 <= s.saving < 1.0
    uni = make_instance(4, 2, C.SPREAD_MAX, 0, 2)
    assert ((uni.stops >= 0) & (uni.stops <= 1)).all()                   # gleichverteilt: alle Stopps im Einheitsquadrat


def test_worst_case_at_the_slider_maximum_finishes_and_is_exact_in_size():
    """Mehr als 12 Stopps gibt es nicht; der größte Fall (4 Spediteure, je 3 Stopps) läuft komplett inklusive Kern und Pool durch (Hänger-Netz, keine Zeit-Assertion)."""
    a = E.analyse(E.Settings(4, 3, 100, 0, 10, 7))
    assert a.solved.inst.m == C.MAX_STOPS
    sav = P.criterion_savings(a.solved)
    assert 0.0 <= sav["zufall"] <= sav["bestmoeglich"] <= a.solved.saving + 1e-12   # abgegebene Stopps (4 von 12) zu verteilen kann nie mehr sparen als alles zu verteilen


def test_same_seed_gives_the_same_instance_in_another_process_with_other_hash_seeds():
    code = "import numpy as np, aal_evaluation as E; a = E.analyse(E.Settings(4, 2, 30, 0, 10, 7)); print(repr(round(a.solved.saving, 12)), a.assignment)"
    outs = []
    for hashseed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True, cwd=ROOT).stdout)
    assert outs[0] == outs[1] and outs[0].strip()


def test_instance_is_deterministic_and_the_seed_matters():
    a, b, c = make_instance(4, 2, 15, 0, 5), make_instance(4, 2, 15, 0, 5), make_instance(4, 2, 15, 0, 6)
    assert np.array_equal(a.stops, b.stops) and np.array_equal(a.depots, b.depots) and not np.array_equal(a.stops, c.stops)


def test_instances_with_different_spreads_share_depots_and_random_numbers():
    """Kopplung der Stufen: gleiche Depots, und die Stopps der engen Stufe sind Depot plus s mal die Zufallszahl (Abstand zum Depot wächst linear mit s)."""
    a, b = make_instance(3, 2, 10, 0, 9), make_instance(3, 2, 20, 0, 9)
    assert np.array_equal(a.depots, b.depots)
    d_a = np.linalg.norm(a.stops - a.depots[list(a.owner)], axis=1)
    d_b = np.linalg.norm(b.stops - b.depots[list(b.owner)], axis=1)
    assert d_b == pytest.approx(2 * d_a)


def test_invalid_instance_is_rejected():
    with pytest.raises(ValueError):
        make_instance(2, 2, 15, 2, 1)                                    # kein Spediteur mit Aufträgen
    with pytest.raises(ValueError):
        make_instance(2, 0, 15, 0, 1)
