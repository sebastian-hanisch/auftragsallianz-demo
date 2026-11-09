"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def _at_with(**values):
    at = _fresh_app()
    for key, value in values.items():
        at.session_state[key] = value
    at.run(timeout=TIMEOUT)
    assert not at.exception, (values, [str(e) for e in at.exception])
    return at


def test_all_minima_and_all_maxima_together_do_not_raise():
    _at_with(n_slider=2, idle_slider=0, per_slider=1, spread_slider=2, ninst_slider=10)
    _at_with(n_slider=5, idle_slider=4, per_slider=6, spread_slider=100, ninst_slider=60)       # idle und per werden auf zulässige Werte begrenzt
    _at_with(n_slider=5, idle_slider=0, per_slider=6, spread_slider=100, ninst_slider=60)        # 5 Spediteure: höchstens 2 Stopps je Spediteur
    _at_with(n_slider=2, idle_slider=1, per_slider=6, spread_slider=100)                          # ein Spediteur mit 6 Stopps


def test_slider_bounds_follow_the_number_of_carriers():
    at = _at_with(n_slider=5, idle_slider=0)
    per = at.sidebar.slider(key="per_slider")
    assert per.max == 2 and per.value <= 2
    idle = at.sidebar.slider(key="idle_slider")
    assert idle.max == 4
    at = _at_with(n_slider=2, idle_slider=1)
    assert at.sidebar.slider(key="per_slider").max == 6


def test_permalink_is_clamped_to_the_slider_bounds():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"], at.query_params["per"], at.query_params["spread"], at.query_params["idle"], at.query_params["ninst"] = "99", "99", "1", "99", "7"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="n_slider").value == 5 and at.sidebar.slider(key="spread_slider").value == 2
    assert at.sidebar.slider(key="idle_slider").value == 4 and at.sidebar.slider(key="per_slider").value == 6 and at.sidebar.slider(key="ninst_slider").value == 10   # nur ein Spediteur mit Aufträgen: bis 6 Stopps


def test_experiment_buttons_show_verdicts_after_the_click():
    at = _at_with(ninst_slider=10)
    assert not [i for i in at.info if "Mittel" in i.value]               # vor dem Klick kein Experiment-Verdikt
    for label, marker in (("🧪 Experiment 1 rechnen", "Das Mittel steigt in"), ("🧪 Experiment 3 rechnen", "geografische"), ("🧪 Experiment 4 rechnen", "Zuwachs je weiterem Mitglied")):
        button = next(b for b in at.button if b.label == label)
        button.click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])
        assert any(marker in i.value for i in at.info), label
    # Experiment 1 rechnet alle Streuungen und bleibt bei einer neuen Streuung gültig; Experiment 3 hängt an der Streuung und verschwindet mit der Bitte um eine neue Rechnung
    at.sidebar.slider(key="spread_slider").set_value(30).run(timeout=TIMEOUT)
    assert any("erneut rechnen" in c.value for c in at.caption)
    assert any("Das Mittel steigt in" in i.value for i in at.info) and not any("geografische" in i.value for i in at.info)


def test_one_stop_per_carrier_has_no_pool_experiment():
    at = _at_with(per_slider=1)
    assert not [b for b in at.button if b.label == "🧪 Experiment 3 rechnen"]
    assert any("nichts zu wählen" in i.value for i in at.info)


def test_the_idle_preset_shows_idle_carriers_in_the_share_table():
    at = _fresh_app()
    at.sidebar.button(key="preset_Nur ein Spediteur hat Aufträge").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.slider(key="idle_slider").value == 3 and at.sidebar.slider(key="per_slider").value == 4
