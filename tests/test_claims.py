"""Jede Zahl der README (Abschnitt „Befunde“) steht hier als Messung mit festen Instanzen (`range(100000, ...)` bzw. die genannten Seed-Bereiche), nie als Einzel-Seed.

Die Zahlen sind Mittel oder Zählungen über die genannte Zahl von Instanzen; Mittel mit Band (Summationsreihenfolge, libm), Zählungen von Ereignissen ohne Toleranzrand exakt,
Zählungen "im Kern" mit Band von 2 (ein Ergebnis auf dem Rand der Toleranz kann zwischen Plattformen kippen)."""

import numpy as np
import pytest

import aal_constants as C
import aal_evaluation as E


def seeds(count, start=100000):
    return range(start, start + count)


@pytest.fixture(scope="module")
def spread_4x2():
    return {r["spread"]: r for r in E.sweep_spread(4, 2, 0, seeds(200))}


@pytest.fixture(scope="module")
def spread_3x3():
    return {r["spread"]: r for r in E.sweep_spread(3, 3, 0, seeds(100))}


@pytest.fixture(scope="module")
def spread_5x2():
    return {r["spread"]: r for r in E.sweep_spread(5, 2, 0, seeds(60))}


def test_saving_over_the_overlap_4_carriers_2_stops_200_instances(spread_4x2):
    expected = {5: (3.67, 0.0, 138), 15: (19.69, 18.57, 16), 30: (33.50, 33.84, 1), 100: (59.47, 60.10, 0)}     # (Mittel, Median, Zahl ohne Ersparnis)
    for spread, (mean, median, zero) in expected.items():
        r = spread_4x2[spread]
        assert r["count"] == 200
        assert r["mean"] == pytest.approx(mean, abs=0.02), spread
        assert r["median"] == pytest.approx(median, abs=0.02), spread
        assert r["zero"] == zero, spread
    assert [spread_4x2[s]["mean"] for s in C.SPREAD_STAGES] == sorted(spread_4x2[s]["mean"] for s in C.SPREAD_STAGES)   # Mittel steigt in allen drei Stufen


def test_saving_over_the_overlap_3_carriers_3_stops_100_instances(spread_3x3):
    expected = {5: (2.08, 84), 15: (14.03, 21), 30: (26.78, 2), 100: (52.82, 0)}
    for spread, (mean, zero) in expected.items():
        assert spread_3x3[spread]["count"] == 100
        assert spread_3x3[spread]["mean"] == pytest.approx(mean, abs=0.02), spread
        assert spread_3x3[spread]["zero"] == zero, spread


def test_saving_over_the_overlap_5_carriers_2_stops_60_instances(spread_5x2):
    expected = {5: (4.48, 34), 15: (21.80, 0), 30: (37.87, 0), 100: (66.74, 0)}
    for spread, (mean, zero) in expected.items():
        assert spread_5x2[spread]["count"] == 60
        assert spread_5x2[spread]["mean"] == pytest.approx(mean, abs=0.02), spread
        assert spread_5x2[spread]["zero"] == zero, spread


def test_the_core_is_never_empty_in_1440_instances(spread_4x2, spread_3x3, spread_5x2):
    total = sum(r["count"] for d in (spread_4x2, spread_3x3, spread_5x2) for r in d.values())
    assert total == 1440
    assert sum(r["core_empty"] for d in (spread_4x2, spread_3x3, spread_5x2) for r in d.values()) == 0


def test_shapley_and_proportional_in_the_core_4_carriers_2_stops(spread_4x2):
    shapley = {5: 200, 15: 164, 30: 151, 100: 180}
    proportional = {5: 139, 15: 22, 30: 48, 100: 137}
    for spread in C.SPREAD_STAGES:
        assert spread_4x2[spread]["shapley_in_core"] == pytest.approx(shapley[spread], abs=2), spread
        assert spread_4x2[spread]["proportional_in_core"] == pytest.approx(proportional[spread], abs=2), spread
    assert sum(r["shapley_in_core"] for r in spread_4x2.values()) < 800            # Shapley liegt nicht immer im Kern


def test_marginal_gain_per_additional_carrier_100_instances():
    rows = E.sweep_members(2, 15, seeds(100))
    assert [r["n"] for r in rows] == [2, 3, 4, 5] and all(r["count"] == 100 for r in rows)
    assert [r["mean"] for r in rows] == pytest.approx([9.49, 13.95, 18.21, 22.09], abs=0.02)
    assert [r["median"] for r in rows] == pytest.approx([0.0, 12.55, 15.88, 21.92], abs=0.02)
    assert [r["delta"] for r in rows[1:]] == pytest.approx([4.46, 4.27, 3.87], abs=0.02)
    assert rows[1]["delta"] > rows[2]["delta"] > rows[3]["delta"]


@pytest.fixture(scope="module")
def pool_15():
    return E.pool_ensemble(4, 3, 15, 0, seeds(60))


@pytest.fixture(scope="module")
def pool_30():
    return E.pool_ensemble(4, 3, 30, 0, seeds(60))


@pytest.fixture(scope="module")
def pool_30_big():
    return E.pool_ensemble(4, 3, 30, 0, seeds(200, 200000))


def _means(cols):
    return {r["criterion"]: r["mean"] for r in E.pool_rows(cols)}


def test_pool_selection_4_carriers_3_stops_spread_15_60_instances(pool_15):
    expected = {"zufall": 6.66, "teuerster": 13.13, "fremdes_depot": 10.45, "fremder_stopp": 9.42, "weit_vom_depot": 12.39, "isoliert": 12.68, "bestmoeglich": 14.81}
    means = _means(pool_15)
    for name, value in expected.items():
        assert means[name] == pytest.approx(value, abs=0.02), name
    wins = {g: E.compare_wins(pool_15, g, "teuerster") for g in C.CRITERIA_GEO}
    assert wins == {"fremdes_depot": (17, 15, 28), "fremder_stopp": (12, 16, 32), "weit_vom_depot": (10, 36, 14), "isoliert": (5, 44, 11)}
    assert all(means["teuerster"] > means[g] for g in C.CRITERIA_GEO)


def test_pool_selection_4_carriers_3_stops_spread_30_60_instances(pool_30):
    expected = {"zufall": 11.04, "teuerster": 19.75, "fremdes_depot": 13.51, "fremder_stopp": 13.40, "weit_vom_depot": 18.08, "isoliert": 19.60, "bestmoeglich": 23.19}
    means = _means(pool_30)
    for name, value in expected.items():
        assert means[name] == pytest.approx(value, abs=0.02), name
    wins = {g: E.compare_wins(pool_30, g, "teuerster") for g in C.CRITERIA_GEO}
    assert wins == {"fremdes_depot": (8, 0, 52), "fremder_stopp": (9, 3, 48), "weit_vom_depot": (11, 23, 26), "isoliert": (11, 39, 10)}
    assert all(means["teuerster"] > means[g] for g in C.CRITERIA_GEO)
    rows = {r["criterion"]: r for r in E.pool_rows(pool_30)}
    assert [rows[c]["optimal"] for c in ("zufall", "teuerster", "fremdes_depot", "fremder_stopp", "weit_vom_depot", "isoliert", "bestmoeglich")] == [0, 13, 3, 4, 8, 12, 60]


def test_pool_selection_spread_30_with_200_other_instances_gives_the_same_order(pool_30_big):
    """Stabilität (DEMO-PLAYBOOK §11a Punkt 5): anderer Seed-Bereich, 200 Instanzen; dieselbe Reihenfolge wie bei 60 Instanzen."""
    means = _means(pool_30_big)
    assert means["teuerster"] > means["isoliert"] > means["weit_vom_depot"] > means["fremdes_depot"] > means["fremder_stopp"] > means["zufall"]
    assert means["bestmoeglich"] > means["teuerster"]
    wins = {g: E.compare_wins(pool_30_big, g, "teuerster") for g in C.CRITERIA_GEO}
    assert wins == {"fremdes_depot": (46, 10, 144), "fremder_stopp": (38, 15, 147), "weit_vom_depot": (25, 102, 73), "isoliert": (38, 108, 54)}
    assert [round(means[c], 1) for c in ("zufall", "teuerster", "fremdes_depot", "fremder_stopp", "weit_vom_depot", "isoliert", "bestmoeglich")] == [11.0, 19.7, 14.2, 14.0, 18.3, 19.2, 22.9]


def test_with_two_stops_per_carrier_the_distance_to_the_depot_rule_equals_the_most_expensive_rule():
    """Bei 2 Stopps je Spediteur wählen 'teuerster' und 'am weitesten vom eigenen Depot' in allen 200 Instanzen dasselbe (Ersparnis gleich); 'isoliert' hat dann einen Gleichstand."""
    cols = E.pool_ensemble(4, 2, 30, 0, seeds(200, 400000))
    assert E.compare_wins(cols, "weit_vom_depot", "teuerster") == (0, 200, 0)


def test_only_one_carrier_has_orders_4_stops_100_instances():
    """Preset „Nur ein Spediteur hat Aufträge“: 4 Spediteure, nur einer mit 4 Stopps, Streuung 30 und 100: Mittel und Zahl ohne Ersparnis."""
    for spread, mean, zero in ((30, 10.60, 34), (100, 24.36, 8)):
        sv = E.ensemble_savings_only(4, 4, spread, 3, seeds(100))
        assert 100 * float(np.mean(sv)) == pytest.approx(mean, abs=0.02), spread
        assert sum(1 for v in sv if v <= E.ZERO) == zero, spread
