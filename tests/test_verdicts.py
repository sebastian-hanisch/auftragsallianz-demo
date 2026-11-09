"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen oder Zeilen, nicht Seeds."""

import aal_constants as C
import aal_evaluation as E


def test_verdict_saving_thresholds_are_inclusive_at_5_and_20_percent():
    assert E.verdict_saving(10.0, 8.0)[0] == "success"                    # genau 20 %
    assert E.verdict_saving(10.0, 8.1)[0] == "info"                       # 19 %
    assert E.verdict_saving(10.0, 9.5)[0] == "info"                       # genau 5 %
    kind, text = E.verdict_saving(10.0, 9.6)                             # 4 %
    assert kind == "info" and "nur **4,0 %**" in text and "(9,60 statt 10,00)" in text


def test_verdict_saving_names_costs_and_percent():
    kind, text = E.verdict_saving(36.0, 4.0)                             # Tausch-Instanz aus conftest: 88,9 %
    assert kind == "success" and "**88,9 %**" in text and "4,00 statt 36,00" in text


def test_verdict_saving_without_saving_says_so():
    for grand in (10.0, 10.0 + 1e-12):
        kind, text = E.verdict_saving(10.0, grand)
        assert kind == "warning" and "spart das Zusammenlegen nichts" in text and "10,00" in text
    assert E.verdict_saving(0.0, 0.0)[0] == "warning"                     # keine Alleinkosten: nichts zu sparen


def _row(spread, mean, median=0.0, zero=0, count=30):
    return {"spread": spread, "mean": mean, "median": median, "zero": zero, "count": count}


def test_verdict_overlap_counts_rising_steps_and_zero_instances():
    rows = [_row(5, 3.7, 0.0, 20), _row(15, 19.7, 18.6, 4), _row(30, 33.5, 33.8, 1), _row(100, 59.5, 60.1, 0)]
    kind, text = E.verdict_overlap(rows)
    assert kind == "info"
    assert "Bei Streuung 5 % spart das Zusammenlegen im Mittel 3,7 %" in text and "(Median 0,0 %, 20 von 30 Instanzen ganz ohne Ersparnis)" in text
    assert "bei Streuung 100 % im Mittel 59,5 % (0 von 30 ohne Ersparnis)" in text and "steigt in 3 von 3 Stufen" in text
    rows[2]["mean"] = 10.0                                                # Knick: nicht mehr überall steigend
    assert "steigt in 2 von 3 Stufen" in E.verdict_overlap(rows)[1]


class _Fake:
    def __init__(self, empty, sh, pr, eps=0.0):
        self.core_empty, self.shapley_in_core, self.proportional_in_core, self.epsilon = empty, sh, pr, eps


def test_verdict_shares_branches():
    assert E.verdict_shares(_Fake(True, False, False, 2.0))[0] == "warning" and "Der Kern ist leer" in E.verdict_shares(_Fake(True, False, False, 2.0))[1]
    kind, text = E.verdict_shares(_Fake(False, True, True))
    assert kind == "success" and "Shapley-Aufteilung im Kern" in text and "proportionale Aufteilung im Kern" in text
    kind, text = E.verdict_shares(_Fake(False, True, False))
    assert kind == "info" and "Shapley-Aufteilung im Kern" in text and "proportionale Aufteilung außerhalb des Kerns" in text


def test_verdict_shares_ensemble_counts_over_all_stages():
    rows = [{"count": 30, "core_empty": 0, "shapley_in_core": 28, "proportional_in_core": 10},
            {"count": 30, "core_empty": 0, "shapley_in_core": 30, "proportional_in_core": 20}]
    kind, text = E.verdict_shares_ensemble(rows)
    assert kind == "info" and "Über alle 60 Instanzen" in text and "in 0 leer" in text and "Shapley-Aufteilung lag in 58" in text and "proportionale in 30" in text
    rows[1]["core_empty"] = 3
    kind, text = E.verdict_shares_ensemble(rows)
    assert kind == "warning" and "in 3 leer" in text


def _pool_rows(teuer, geo, zufall=5.0, best=15.0):
    return [{"criterion": "teuerster", "mean": teuer, "count": 60}, {"criterion": "fremdes_depot", "mean": geo, "count": 60},
            {"criterion": "zufall", "mean": zufall, "count": 60}, {"criterion": "bestmoeglich", "mean": best, "count": 60}]


def test_verdict_pool_three_branches_and_counts():
    kind, text = E.verdict_pool(_pool_rows(13.1, 10.4), (17, 15, 28), "fremdes_depot")
    assert kind == "info" and text.startswith("Hier liegt der teuerste Auftrag im Mittel vor jedem geografischen Kriterium")
    assert "im Mittel 10,4 %, teuerster Auftrag 13,1 %, Zufall 5,0 %, bestmöglich 15,0 % (60 Instanzen)" in text
    assert "in 17 Instanzen besser, in 15 gleich und in 28 schlechter" in text
    assert E.verdict_pool(_pool_rows(10.0, 12.0), (1, 2, 3), "fremdes_depot")[1].startswith("Hier liegt ein geografisches Kriterium vorn")
    assert E.verdict_pool(_pool_rows(10.0, 10.0), (1, 2, 3), "fremdes_depot")[1].startswith("Beide liegen gleichauf")


def test_verdict_pool_small_gap_is_called_nearly_equal():
    text = E.verdict_pool(_pool_rows(13.1, 12.6), (5, 44, 11), "fremdes_depot")[1]
    assert text.startswith("Hier liegt der teuerste Auftrag nur knapp vor dem besten geografischen Kriterium") and "nahezu gleichauf" in text


def test_best_geo_name_picks_the_largest_mean_and_the_first_on_ties():
    rows = [{"criterion": c, "mean": m} for c, m in zip(C.CRITERIA_GEO, (1.0, 3.0, 3.0, 2.0))]
    assert E.best_geo_name(rows) == C.CRITERIA_GEO[1]


def test_compare_wins_counts_with_the_tolerance():
    cols = {"a": [0.3, 0.1, 0.2, 0.5], "b": [0.1, 0.1 + 1e-12, 0.4, 0.5]}
    assert E.compare_wins(cols, "a", "b") == (1, 2, 1)


def test_verdict_members_names_first_last_and_increments():
    rows = [{"n": 2, "count": 30, "mean": 9.5, "median": 0.0, "delta": None}, {"n": 3, "count": 30, "mean": 13.9, "median": 12.5, "delta": 4.4},
            {"n": 4, "count": 30, "mean": 18.2, "median": 15.9, "delta": 4.3}, {"n": 5, "count": 30, "mean": 22.1, "median": 21.9, "delta": 3.9}]
    kind, text = E.verdict_members(rows)
    assert kind == "info" and "Bei 2 Spediteuren spart das Zusammenlegen im Mittel 9,5 %, bei 5 sind es 22,1 % (30 Instanzen)" in text
    assert "Prozentpunkten: +4,4, +4,3, +3,9" in text and "sinkt in 2 von 2 Schritten" in text


def test_saving_percent_has_a_zero_branch():
    assert E.saving_percent(1.0, 0.0) == 0.0 and E.saving_percent(4.0, 36.0) == 100 * (1 - 4 / 36)
