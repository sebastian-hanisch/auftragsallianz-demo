"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und jedes Preset durchläuft die Kernfrage mit dem erwarteten Verdikt
(DEMO-PLAYBOOK §11a Punkt 7: ein Preset ohne Effekt meldet keinen Effekt)."""

import pytest

import aal_constants as C
import aal_evaluation as E
import aal_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and 0 <= p["idle"] <= p["n"] - 1 and C.PER_MIN <= p["per"] <= C.max_per(p["n"], p["idle"]), name
        assert C.SPREAD_MIN <= p["spread"] <= C.SPREAD_MAX and C.NINST_MIN <= p["ninst"] <= C.NINST_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert E.Settings(**C.PRESETS["Mittlere Streuung (Voreinstellung)"]) == E.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_effective_settings_leave_every_preset_unchanged():
    for name, p in C.PRESETS.items():
        assert E.effective_settings(E.Settings(**p)) == E.Settings(**p), name


EXPECTED = {   # Preset: (Ersparnis in %, Art des Verdikts, Kern leer?, Shapley im Kern?, proportional im Kern?); gemessen mit dem Code der Demo
    "Mittlere Streuung (Voreinstellung)": (14.1, "info", False, True, False),
    "Eng um die Depots": (0.0, "warning", False, True, True),
    "Gleichverteilt im Gebiet": (46.3, "success", False, True, True),
    "Fünf Spediteure": (12.6, "info", False, False, False),
    "Nur ein Spediteur hat Aufträge": (13.5, "info", False, True, True),
}


def test_expected_table_covers_every_preset():
    assert set(EXPECTED) == set(C.PRESETS)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_through_the_core_question_with_the_expected_verdict(name):
    saving, kind_expected, empty, sh_in, pr_in = EXPECTED[name]
    a = E.analyse(E.Settings(**C.PRESETS[name]))
    pct = E.saving_percent(a.solved.grand, float(a.solved.alone.sum()))
    assert pct == pytest.approx(saving, abs=0.05)
    kind, text = E.verdict_saving(float(a.solved.alone.sum()), a.solved.grand)
    assert kind == kind_expected
    assert (a.core_empty, a.shapley_in_core, a.proportional_in_core) == (empty, sh_in, pr_in)
    if kind == "warning":
        assert "nichts" in text and "%" not in text                    # kein Effekt: keine Prozentzahl als Befund
    # Experiment-Verdikte für das Preset laufen mit wenigen Instanzen ohne Fehler und mit Text
    s = a.settings
    rows = E.sweep_spread(s.n, s.per, s.idle, range(100000, 100003))
    assert E.verdict_overlap(rows)[1] and E.verdict_shares_ensemble(rows)[1] and E.verdict_shares(a)[1]
    if s.per > 1:
        cols = E.pool_ensemble(s.n, s.per, s.spread, s.idle, range(100000, 100003))
        rows3 = E.pool_rows(cols)
        assert E.verdict_pool(rows3, E.compare_wins(cols, E.best_geo_name(rows3), "teuerster"), E.best_geo_name(rows3))[1]


def test_the_idle_preset_has_idle_carriers_that_the_shapley_rule_pays():
    a = E.analyse(E.Settings(**C.PRESETS["Nur ein Spediteur hat Aufträge"]))
    assert a.solved.alone[1:].tolist() == [0.0, 0.0, 0.0] and (a.shapley[1:] <= 1e-12).all() and (a.shapley[1:] < -1e-3).any()
    assert a.proportional[1:].tolist() == [0.0, 0.0, 0.0]


def test_the_closest_preset_has_no_saving_and_a_nonempty_core():
    a = E.analyse(E.Settings(**C.PRESETS["Eng um die Depots"]))
    assert a.solved.saving <= E.ZERO and not a.core_empty
