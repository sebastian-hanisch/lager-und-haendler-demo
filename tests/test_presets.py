"""Presets: Vollständigkeit, gültige Werte auf gemessenen Zellen, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen und Einrasten)."""

import pytest

import lah_constants as C
import lah_evaluation as ev
import lah_presets as P
import lah_results as R


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_lie_on_measured_cells_and_inside_the_bounds():
    for name, preset in C.PRESETS.items():
        assert preset["lr"] in C.LEADS and preset["lw"] in C.LEADS and preset["hw"] in C.H_W_GRID and preset["p"] in C.P_GRID and preset["view"] in C.VIEWS
        assert C.D1_MIN <= preset["d1"] <= C.D1_MAX and C.D2_MIN <= preset["d2"] <= C.D2_MAX
        R.get_cell(preset["lr"], preset["lw"], preset["hw"], preset["p"])       # wirft, wenn die Zelle nicht gemessen ist


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert ev.Settings(**{k: first[k] for k in ("lr", "lw", "hw", "p", "view", "d1", "d2")}) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
    assert {k for k, s in P.SETTING_SPECS.items() if s.scenario} == set(P.PRESET_KEYS.values())     # Anzeigeparameter sind genau Seed und Länge
    assert {k for k, s in P.SETTING_SPECS.items() if not s.scenario} == {"seed_input", "show_slider"}


def test_preset_names_say_what_the_values_are():
    """Beschriftungen gegen die tatsächlichen Werte (Klasse 'vertauschte Beschriftungen')."""
    p = C.PRESETS
    assert (p["Hauptfall (1/1, h_w 0,4, p 9)"]["lr"], p["Hauptfall (1/1, h_w 0,4, p 9)"]["lw"], p["Hauptfall (1/1, h_w 0,4, p 9)"]["hw"], p["Hauptfall (1/1, h_w 0,4, p 9)"]["p"]) == (1, 1, 0.4, 9)
    assert p["Langer Weg zum Lager (1/2)"]["lw"] == 2 and p["Langer Weg zum Lager (1/2)"]["lr"] == 1
    assert p["Langer Weg zum Händler (2/1)"]["lr"] == 2 and p["Langer Weg zum Händler (2/1)"]["lw"] == 1
    assert p["Teures Lager (h_w 0,8)"]["hw"] == 0.8 and p["Billiges Lager (h_w 0,2)"]["hw"] == 0.2
    assert p["Hohe Fehlmengenkosten (p 19)"]["p"] == 19


def test_coerce_clamps_numbers_and_snaps_only_allowed_choices():
    assert P.coerce(P.SETTING_SPECS["d1_slider"], "99") == C.D1_MAX
    assert P.coerce(P.SETTING_SPECS["d2_slider"], "-99") == C.D2_MIN
    assert P.coerce(P.SETTING_SPECS["hw_select"], "0.6") == 0.6
    assert P.coerce(P.SETTING_SPECS["hw_select"], "0.5") is None         # nicht gemessen: ignorieren, nicht die nächste Zelle nehmen
    assert P.coerce(P.SETTING_SPECS["lr_radio"], "3") is None
    assert P.coerce(P.SETTING_SPECS["view_radio"], "C") == "C" and P.coerce(P.SETTING_SPECS["view_radio"], "D") is None
    with pytest.raises(ValueError):
        P.coerce(P.SETTING_SPECS["p_select"], "abc")


def test_every_combination_of_the_choice_controls_is_a_measured_cell():
    for lr in C.LEADS:
        for lw in C.LEADS:
            for hw in C.H_W_GRID:
                for p in C.P_GRID:
                    assert R.get_cell(lr, lw, hw, p)["Lr"] == lr
    with pytest.raises(KeyError):
        R.get_cell(1, 1, 0.5, 9)
