"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Die Regler Lieferzeit, h_w, p und Sicht liegen auf gemessenen Werten (Auswahl statt freier Zahl); ein Permalink mit einem anderen Wert wird ignoriert.
Zufalls-Seed und Länge des Verlaufs gehören nicht zum Szenario: ihre Änderung hebt das aktive Preset nicht auf (`scenario=False`)."""

import random
from dataclasses import dataclass

import streamlit as st

import lah_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None     # Auswahlregler: erlaubte Werte
    scenario: bool = True            # False: Anzeigeparameter, ändert das Szenario nicht


SETTING_SPECS = {
    "lr_radio": SettingSpec("lr", int, C.DEFAULT_LR, options=C.LEADS),
    "lw_radio": SettingSpec("lw", int, C.DEFAULT_LW, options=C.LEADS),
    "hw_select": SettingSpec("hw", float, C.DEFAULT_HW, options=C.H_W_GRID),
    "p_select": SettingSpec("p", int, C.DEFAULT_P, options=C.P_GRID),
    "view_radio": SettingSpec("view", str, C.DEFAULT_VIEW, options=C.VIEWS),
    "d1_slider": SettingSpec("d1", int, 0, C.D1_MIN, C.D1_MAX),
    "d2_slider": SettingSpec("d2", int, 0, C.D2_MIN, C.D2_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, scenario=False),
    "show_slider": SettingSpec("show", int, C.DEFAULT_SHOW, C.SHOW_MIN, C.SHOW_MAX, scenario=False),
}
PRESET_KEYS = {"lr": "lr_radio", "lw": "lw_radio", "hw": "hw_select", "p": "p_select", "view": "view_radio", "d1": "d1_slider", "d2": "d2_slider"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def options(state_key):
    return list(SETTING_SPECS[state_key].options)


def coerce(spec: SettingSpec, raw):
    """Wert aus der Adresszeile in den Regler: Zahlen auf die Grenzen begrenzt, Auswahlwerte nur, wenn sie erlaubt sind (sonst None)."""
    value = spec.caster(raw)
    if spec.options is not None:
        return value if value in spec.options else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = coerce(spec, qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen des Szenarios entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Szenario-Regler aus SETTING_SPECS (ein Preset ohne Eintrag meint den Standardwert) mit dem Session State."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items() if spec.scenario
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
