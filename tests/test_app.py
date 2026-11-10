"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py mit den Standardeinstellungen, durchläuft jedes Preset, jede Lieferzeit-Kombination, jede Sicht, jeden Regler an
seinen Grenzen und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei Diagrammen und Abstürze bei Auswahlreglern)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import lah_constants as C
import lah_evaluation as ev
import lah_presets as P
from lah_pdf_export import build_pdf

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last (parallele Läufe) reichen 180 s nicht immer


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _expander(at, label):
    return next(e for e in at.expander if e.label == label)         # nie per Index: neue Expander verschieben ihn


def test_app_loads_without_exception_and_has_no_scaffold_text():
    at = _fresh_app()
    texts = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "TODO" not in texts and "Gerüst" not in texts
    assert at.title[0].value.startswith("🏬 Lager und Händler")
    assert {"⚠️ Annahmen, Grenzen und bekannte Probleme", "📐 Mathematische Formulierung"} <= {e.label for e in at.expander}


def test_default_cell_shows_the_decomposition_verdict_and_three_metrics():
    at = _fresh_app()
    assert len(at.metric) == 3
    assert any("Kein Unterschied auf fünf Stellen" in s.value and "S1 = 10, S2 = 13" in s.value for s in at.success)
    assert at.metric[0].value == "6,6013"                       # gemeinsame DP, Hauptzelle (6,60129 auf fünf Stellen)
    assert any("Sicht A" in w.value and "+11,51 %" in w.value for w in at.warning)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, (name, [str(e) for e in at.exception])
    preset = C.PRESETS[name]
    assert (at.sidebar.radio(key="lr_radio").value, at.sidebar.radio(key="lw_radio").value) == (preset["lr"], preset["lw"])
    assert at.sidebar.select_slider(key="hw_select").value == preset["hw"] and at.sidebar.select_slider(key="p_select").value == preset["p"]


@pytest.mark.parametrize("lr,lw", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_every_lead_time_combination_runs_with_every_view(lr, lw):
    at = _fresh_app()
    at.sidebar.radio(key="lr_radio").set_value(lr)
    at.sidebar.radio(key="lw_radio").set_value(lw).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for view in C.VIEWS:
        at.sidebar.radio(key="view_radio").set_value(view).run(timeout=TIMEOUT)
        assert not at.exception, (lr, lw, view, [str(e) for e in at.exception])


def test_every_choice_value_runs():
    at = _fresh_app()
    for h in C.H_W_GRID:
        at.sidebar.select_slider(key="hw_select").set_value(h).run(timeout=TIMEOUT)
        assert not at.exception, (h, [str(e) for e in at.exception])
    for p in C.P_GRID:
        at.sidebar.select_slider(key="p_select").set_value(p).run(timeout=TIMEOUT)
        assert not at.exception, (p, [str(e) for e in at.exception])


@pytest.mark.parametrize("key", ["d1_slider", "d2_slider", "show_slider"])
def test_every_slider_at_its_min_and_max_does_not_raise(key):
    for edge in ("min", "max"):
        at = _fresh_app()
        slider = at.sidebar.slider(key=key)
        slider.set_value(int(slider.min if edge == "min" else slider.max)).run(timeout=TIMEOUT)
        assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_seed_extremes_and_the_new_run_button():
    at = _fresh_app()
    for seed in (0, C.SEED_MAX):
        at.sidebar.number_input(key="seed_input").set_value(seed).run(timeout=TIMEOUT)
        assert not at.exception
    old = at.sidebar.number_input(key="seed_input").value
    next(b for b in at.sidebar.button if b.label.startswith("🎲")).click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.number_input(key="seed_input").value != old


def test_level_offsets_change_the_verdict_text():
    at = _fresh_app()
    at.sidebar.slider(key="d2_slider").set_value(3).run(timeout=TIMEOUT)
    assert any("S2 = 16" in i.value and "gegenüber den Niveaus der Zerlegung" in i.value for i in at.info)


def test_permalink_clamps_numbers_and_ignores_unmeasured_values():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["lr"] = "2"
    at.query_params["hw"] = "0.5"          # nicht gemessen: ignorieren
    at.query_params["d1"] = "99"           # auf die Grenze begrenzen
    at.query_params["view"] = "C"
    at.run()
    assert not at.exception
    assert at.sidebar.radio(key="lr_radio").value == 2
    assert at.sidebar.select_slider(key="hw_select").value == C.DEFAULT_HW
    assert at.sidebar.slider(key="d1_slider").value == C.D1_MAX
    assert at.sidebar.radio(key="view_radio").value == "C"


def test_hand_example_section_shows_the_hand_numbers():
    at = _fresh_app()
    text = " ".join(m.value for m in at.markdown)
    assert "S1 = 3" in text and "S2 = 4" in text and "31/16" in text and "(1, 4, 6, 4, 1)/16" in text


def test_dp_button_is_only_offered_for_lead_time_2_2_and_runs_there():
    at = _fresh_app()
    assert not [b for b in at.main.button if b.key == "dp_button"]       # 1/1: die App rechnet selbst
    assert any("Gemeinsame DP eben gerechnet" in s.value for s in at.success)


@pytest.mark.slow
def test_dp_button_for_lead_time_2_2_computes_the_joint_dp():
    """Rund 200.000 Zustände: Größenordnung zehn Sekunden (lastbehaftet), deshalb als langsam markiert."""
    at = _fresh_app()
    at.sidebar.radio(key="lr_radio").set_value(2)
    at.sidebar.radio(key="lw_radio").set_value(2).run(timeout=TIMEOUT)
    button = next(b for b in at.main.button if b.key == "dp_button")
    button.click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Gemeinsame DP eben gerechnet" in s.value and "9,14229" in s.value for s in at.success)


def test_pdf_export_is_a_pdf():
    settings = ev.Settings()
    data = build_pdf(settings, ev.analyse(settings))
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_every_preset_cell_is_covered_by_the_state_keys():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
