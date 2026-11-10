"""Lager und Händler: Zerlegung statt gemeinsamer Rechnung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück der Reihe „Bestandspolitiken“: zwei Bestände in Reihe (Lager beliefert Händler), die optimale Politik als Zerlegung in zwei eindimensionale
Rechnungen, verglichen mit der gemeinsamen Rückwärtsrechnung und mit der lokalen Regel in drei Sichten des Lagers.

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import pandas as pd
import streamlit as st

import lah_constants as C
import lah_evaluation as ev
import lah_example as EX
import lah_presets as PR
import lah_results as R
import lah_scaling as SC
import lah_visualization as V
from lah_evaluation import de, pct
from lah_layout import dunkles_design, kopf_css
from lah_pdf_export import build_pdf
from lah_presets import init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params

st.set_page_config(page_title="Lager und Händler – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _run(settings):
    return ev.run(settings)


@st.cache_data(show_spinner=False)
def _scaling():
    return SC.table()


@st.cache_data(show_spinner=False)
def _example():
    return EX.report()


@st.cache_data(show_spinner=False)
def _dp_live(lr, lw, hw, p):
    return ev.solve_dp_live(R.get_cell(lr, lw, hw, p))


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lah_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🏬 Lager und Händler: Zerlegung statt gemeinsamer Rechnung")
st.markdown(
    """
Ein **Händler** bedient die Kunden und bestellt bei einem **Lager**, das beim Lieferanten bestellt. Mit zwei gekoppelten Beständen wächst der Zustand der gemeinsamen
Rechnung mit jeder zusätzlichen Periode Lieferzeit auf das 14,6- bis 16,9-Fache (bei 14 Werten je Pipelineplatz; Rechenlast-Tafel unten, Raster der Messung). Die **Kernfrage** dieser Demo: **Reicht es, jede Stufe für sich zu steuern?**
Im Modell der Demo (Poisson-Nachfrage, Rückstand beim Händler, feste Lieferzeiten, Bestellgrenze 13) ist die optimale Politik ein **Base-Stock auf dem Echelon-Bestand**, und ihre zwei Niveaus
folgen aus **zwei eindimensionalen Rechnungen** statt aus einer gemeinsamen. Was „jede Stufe für sich“ kostet, hängt davon ab, **was das Lager vom Händler sieht**.

Weiter unten: die **Kernfrage mit den zwei Rechnungen**, vier Experimente (**Verlauf**, **Abhängigkeit der Lücke**, **Informationsverzug**, **Rechenlast**), das **Handbeispiel von Hand**
und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Zerlegung", expanded=True):
    st.markdown(
        """
1. **Echelon-Bestand:** der Echelon-Bestand einer Stufe ist ihr Bestand plus alles, was hinter ihr liegt. Beim Lager sind das der Lagerbestand, die Ware auf dem Weg zum Händler und der Händlerbestand.
2. **Schritt 1, Händlerstufe:** eine eindimensionale Rechnung über die Nachfrage der Lieferzeit des Händlers plus einer Periode liefert das Niveau **S1** (Minimum von L1).
3. **Schritt 2, Lagerstufe:** eine zweite eindimensionale Rechnung nimmt die Kostenfunktion der Händlerstufe als gegeben (mit dem Niveau S1 als Deckel) und liefert das Niveau **S2** (Minimum von G2). Die Rechnungen laufen nacheinander, nicht unabhängig.
4. **Politik:** der Händler bestellt bis S1 auf seiner Bestandsposition, das Lager bestellt bis S2 auf dem Echelon-Bestand. Die gemeinsame Rückwärtsrechnung ergibt in allen gemessenen Zellen (Bestellgrenze 13) dieselben Kosten.
5. **Lokale Regel:** bestellt das Lager nur nach seinem eigenen Bestand, verliert es im Modell der Demo, weil ihm die Sicht auf die Bestellungen des Händlers fehlt (Vergleich der Sichten A, B, C). Wie viel, zeigt der Vergleich in drei Sichten.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Lieferzeiten (Perioden)**")
    lr = st.radio("Lieferzeit Händler beim Lager", PR.options("lr_radio"), key="lr_radio", horizontal=True,
                  help="Wie viele Perioden eine Bestellung des Händlers beim Lager unterwegs ist.")
    lw = st.radio("Lieferzeit Lager beim Lieferanten", PR.options("lw_radio"), key="lw_radio", horizontal=True,
                  help="Wie viele Perioden eine Bestellung des Lagers beim Lieferanten unterwegs ist.")
    st.markdown("**Kosten**")
    hw = st.select_slider("Lagerkostensatz h_w", PR.options("hw_select"), key="hw_select", format_func=lambda v: de(v, 1),
                          help="Kosten je Stück und Periode im Lager und auf dem Weg zum Händler. Der Händler zahlt 1,0 je Stück und Periode für seinen Bestand (insgesamt).")
    p = st.select_slider("Fehlmengenkosten p", PR.options("p_select"), key="p_select",
                         help="Kosten je fehlendem Stück beim Händler (Rückstand wird nachgeliefert, die Nachfrage geht nicht verloren).")
    st.markdown("**Was sieht das Lager?**")
    view = st.radio("Sicht des Lagers (lokale Regel)", PR.options("view_radio"), key="view_radio", format_func=lambda v: C.VIEW_LABEL[v], help=C.VIEW_HELP)
    st.markdown("**Niveaus der Echelon-Politik**")
    d1 = st.slider("Abweichung von S1", *PR.bounds("d1_slider"), key="d1_slider", help="0 = Niveau der Zerlegung. Wirkt auf die Echelon-Politik (Kennzahl in der Kernfrage, Verlauf).")
    d2 = st.slider("Abweichung von S2", *PR.bounds("d2_slider"), key="d2_slider", help="0 = Niveau der Zerlegung.")
    st.markdown("**Verlauf**")
    seed = st.number_input("Zufalls-Seed des Nachfrageverlaufs", *PR.bounds("seed_input"), key="seed_input", step=1)
    show = st.slider("Gezeigte Perioden", *PR.bounds("show_slider"), key="show_slider")
    st.button("🎲 Neuen Nachfrageverlauf erzeugen", width="stretch", on_click=randomize_seed)

sync_query_params({
    "lr_radio": lr, "lw_radio": lw, "hw_select": hw, "p_select": p, "view_radio": view, "d1_slider": d1, "d2_slider": d2, "seed_input": int(seed), "show_slider": int(show),
})

settings = ev.Settings(lr=int(lr), lw=int(lw), hw=float(hw), p=int(p), view=view, d1=int(d1), d2=int(d2), seed=int(seed), show=int(show))
a = _analysis(settings)
cell = a.cell
rule = cell["rules"][view]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(
    f"Zelle: Lieferzeit Händler/Lager {lr}/{lw}, h_w {de(hw, 1)}, p {p}; Poisson-Nachfrage Mittel 3, Bestellgrenze {C.Q_MAX} je Stufe, Kosten je Periode im Mittel. "
    "Die Werte dieser Zelle sind vorgerechnet (`data/lah_grid.json`, erzeugt von `tools/build_grid.py`)."
)
m1, m2, m3 = st.columns(3)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Gemeinsame DP (Optimum)", de(cell["g_dp"], 4), delta="Referenz", delta_color="off")
m2.metric("Echelon-Politik (Zerlegung)", de(cell["g_echelon"], 4), delta=f"{de(cell['g_echelon'] - cell['g_dp'], 4, sign=True)} gegen DP", delta_color="inverse")
m3.metric(f"Lokal, Sicht {view}", de(rule["g"], 4), delta=f"{de(rule['g'] - cell['g_dp'], 4, sign=True)} gegen DP ({pct(a.gaps[view], 2, sign=True)})", delta_color="inverse")
st.plotly_chart(V.build_rule_costs(cell), width="stretch", key="rule_costs")
st.caption(
    "Kosten je Periode nach Anteilen. Echelon-Politik und gemeinsame DP sind in dieser Zelle gleich (eine Säule). Bei den lokalen Regeln sind die Niveaus per Bergsteigen gesucht: "
    f"A (S1 = {cell['rules']['A']['S1']}, Sw = {cell['rules']['A']['S2']}), B (S1 = {cell['rules']['B']['S1']}, Sw = {cell['rules']['B']['S2']}), "
    f"C (S1 = {cell['rules']['C']['S1']}, Sw = {cell['rules']['C']['S2']}); Sw ist das Niveau des Lagers auf dem Bestand, den es in seiner Sicht sieht."
)
st.download_button("📄 Bericht als PDF", data=build_pdf(settings, a), file_name="lager-und-haendler.pdf", mime="application/pdf")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.markdown("## ❓ Kernfrage: Reicht es, jede Stufe für sich zu steuern?")
tally = R.echelon_equals_dp()
st.markdown("**1. Zerlegung gegen gemeinsame Rechnung**")
kind, text = ev.verdict_decomposition(cell["g_dp"], cell["g_echelon"], cell["S1"], cell["S2"], tally["cost_equal"], tally["n"])
getattr(st, kind)(text)
st.plotly_chart(V.build_decomposition(ev.params_of(settings), a.dec, a.S1, a.S2), width="stretch", key="decomposition")
st.caption(
    "Links die Händlerstufe: Kosten L1(y) in Abhängigkeit vom Echelon-Bestand y nach der Bestellung; ihr Minimum ist S1. Rechts die Lagerstufe: G2(y) mit der von der Händlerstufe "
    "induzierten Kostenfunktion (der Bestand des Händlers ist höchstens S1); ihr Minimum ist S2 und zugleich die mittleren Kosten der Politik (geschlossene Formel, ohne Bestellgrenze)."
)
kind, text = ev.verdict_levels(d1, d2, a.S1, a.S2, a.g_levels, a.dec.g)
getattr(st, kind)(text)

if settings.lr + settings.lw <= C.DP_AUTO_MAX_SUM:
    with st.spinner("Gemeinsame DP wird gerechnet (unter einer Sekunde)…"):
        live = _dp_live(settings.lr, settings.lw, settings.hw, settings.p)
else:
    live = None
    st.caption("Bei Lieferzeit 2/2 hat die gemeinsame Rechnung rund 200.000 Zustände; sie läuft auf Knopfdruck (Größenordnung zehn Sekunden, auf einem Server länger).")
    key = (settings.lr, settings.lw, settings.hw, settings.p)
    if st.button("▶ Gemeinsame DP jetzt rechnen (etwa 10 s)", key="dp_button"):
        with st.spinner("Gemeinsame DP auf rund 200.000 Zuständen, Größenordnung zehn Sekunden…"):
            st.session_state["dp_live"] = (key, _dp_live(*key))
    stored = st.session_state.get("dp_live")
    if stored is not None and stored[0] == key:
        live = stored[1]
if live is not None:
    kind, text = ev.verdict_dp_live(live[0], cell["g_dp"], live[2], live[1])
    getattr(st, kind)(text)

st.markdown("**2. Lokale Regel: drei Sichten des Lagers**")
kind, text = ev.verdict_view(view, a.gaps[view], *(lambda pr: (sum(q["diff"] < ev.EQUAL_TOL for q in pr), len(pr)))(R.delay_pairs()))
getattr(st, kind)(text)
st.dataframe(
    pd.DataFrame([
        dict(Sicht=C.VIEW_LABEL[k], Niveaus=f"S1 = {cell['rules'][k]['S1']}, Sw = {cell['rules'][k]['S2']}", Kosten=cell["rules"][k]["g"], Lücke=a.gaps[k])
        for k in C.VIEWS
    ]),
    hide_index=True, width="stretch",
    column_config={"Kosten": st.column_config.NumberColumn("Kosten je Periode", format="%.5f"), "Lücke": st.column_config.NumberColumn("Lücke zur DP", format="%+.2f %%")},
)
st.caption(
    "Die Lücke ist auf die Kosten der gemeinsamen DP bezogen. Ohne Nennung der Sicht ist „die lokale Regel verliert x %“ nicht lesbar: "
    f"in dieser Zelle {pct(a.gaps['A'], 2, sign=True)} (A), {pct(a.gaps['B'], 2, sign=True)} (B), {pct(a.gaps['C'], 2, sign=True)} (C)."
)

# --- Experiment 1: Verlauf -----------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment: Verlauf über denselben Nachfrageverlauf")
with st.spinner("Verlauf wird simuliert (20.000 Perioden, einige Zehntelsekunden)…"):
    run = _run(settings)
st.plotly_chart(V.build_trace(run, view, settings.show), width="stretch", key="trace")
st.caption(
    f"Echelon-Politik (S1 = {a.S1}, S2 = {a.S2}) gegen die lokale Regel in Sicht {view} (S1 = {rule['S1']}, Sw = {rule['S2']}) über denselben Nachfragestrom "
    f"(Zufalls-Seed {settings.seed}, {settings.show} Perioden nach {C.BURN_IN} Einschwingperioden; ein leerer Start würde die Kosten verfälschen)."
)
kind, text = ev.verdict_lag(view, run.lag_echelon, run.lag_local)
getattr(st, kind)(text)
st.dataframe(
    pd.DataFrame([
        {"Regel": "Echelon-Politik", **{f"= Nachfrage von t − {k}": 100 * v for k, v in run.lag_echelon.items()}},
        {"Regel": f"lokal, Sicht {view}", **{f"= Nachfrage von t − {k}": 100 * v for k, v in run.lag_local.items()}},
    ]),
    hide_index=True, width="stretch",
    column_config={f"= Nachfrage von t − {k}": st.column_config.NumberColumn(format="%.1f %%") for k in run.lag_echelon},
)
st.caption(
    f"Anteil der Perioden, in denen die Bestellung des Lagers genau der Nachfrage von t − 1, t − 2, t − 3 entspricht ({C.RUN_LENGTH:,} Perioden nach der Einschwingzeit).".replace(",", ".")
)
e_rule, l_rule = cell["rules"]["echelon"], rule
st.dataframe(
    pd.DataFrame([
        {"Kennzahl": "Mittlere Kosten je Periode, exakt", "Echelon-Politik": e_rule["g"], f"lokal {view}": l_rule["g"]},
        {"Kennzahl": "Mittlerer Lagerbestand, exakt", "Echelon-Politik": e_rule["mean_w"], f"lokal {view}": l_rule["mean_w"]},
        {"Kennzahl": "Anteil der Zeit, in dem das Lager die Händlerbestellung nicht voll bedienen kann (%)", "Echelon-Politik": 100 * e_rule["unfilled"], f"lokal {view}": 100 * l_rule["unfilled"]},
        {"Kennzahl": f"Mittlere Kosten im gezeigten Lauf ({C.RUN_LENGTH:,} Perioden), simuliert".replace(",", "."), "Echelon-Politik": run.mean_echelon, f"lokal {view}": run.mean_local},
    ]),
    hide_index=True, width="stretch",
    column_config={"Echelon-Politik": st.column_config.NumberColumn(format="%.3f"), f"lokal {view}": st.column_config.NumberColumn(format="%.3f")},
)
more, total = R.warehouse_stock_larger()
st.caption(
    f"Exakte Werte aus der stationären Verteilung der Kette (vorgerechnet); der Lauf oben ist eine Simulation und streut um sie. Im Mittel hält die Regel A in {more} von {total} Zellen mehr Lagerbestand "
    "als die Echelon-Politik; das lässt sich als Mehrbestand deuten, der die fehlende Sicht ersetzt (Deutung, nicht als Ursache gemessen)."
)

# --- Experiment 2: Lücke ------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment: Wovon hängt die Lücke der lokalen Regel ab?")
rows = R.cell_rows()
st.plotly_chart(V.build_gap_lines(rows, view, settings.lr, settings.lw, settings.hw, settings.p), width="stretch", key="gap_lines")
hw_ok, hw_n = R.monotone_in_hw(view)
p_ok, p_n, p_exc = R.monotone_in_p(view)
kind, text = ev.verdict_trend(view, hw_ok, hw_n, p_ok, p_n, [(k[0], k[1], k[2]) for k in p_exc])
getattr(st, kind)(text)
sel = [r for r in rows if r["Lr"] == settings.lr and r["Lw"] == settings.lw]
st.dataframe(
    pd.DataFrame(sel).rename(columns={"h_w": "h_w", "g_dp": "DP-Kosten", "gap_A": "Lücke A", "gap_B": "Lücke B", "gap_C": "Lücke C"}).drop(columns=["Lr", "Lw"]),
    hide_index=True, width="stretch",
    column_config={
        "DP-Kosten": st.column_config.NumberColumn(format="%.5f"),
        "Lücke A": st.column_config.NumberColumn(format="%+.2f %%"), "Lücke B": st.column_config.NumberColumn(format="%+.2f %%"), "Lücke C": st.column_config.NumberColumn(format="%+.2f %%"),
    },
)
lr_ok, lr_n, _ = R.lead_effect(view, "Lr")
lw_ok, lw_n, lw_exc = R.lead_effect(view, "Lw")
lo, hi = R.gap_range(view)
st.caption(
    f"Alle 48 Zellen, Sicht {view}: Lücke {pct(lo, 2, sign=True)} bis {pct(hi, 2, sign=True)}. Bei Lieferzeit 2 des Händlers ist sie kleiner als bei 1 in {lr_ok} von {lr_n} Paaren, "
    f"bei Lieferzeit 2 des Lagers kleiner als bei 1 in {lw_ok} von {lw_n} Paaren (Ausnahmen: {len(lw_exc)}). Die Tabelle zeigt die Zellen der gewählten Lieferzeiten."
)

# --- Experiment 3: Informationsverzug -------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment: Der Verlust wirkt wie Informationsverzug")
pairs = R.delay_pairs()
st.plotly_chart(V.build_delay(pairs), width="stretch", key="delay")
kind, text = ev.verdict_delay(pairs)
getattr(st, kind)(text)
st.caption(
    "Je Punkt eine Zelle (Lieferzeit Händler, h_w, p): x = Kosten der Regel A bei Lagerlieferzeit 1, y = Kosten der Regel B bei Lagerlieferzeit 2, jeweils mit den besten Niveaus der Regel. "
    "Das passt zur Deutung, dass dem Lager bei Sicht A eine Periode Information fehlt; bewiesen ist die Ursache damit nicht."
)

# --- Experiment 4: Rechenlast ---------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment: Rechenlast der gemeinsamen Rechnung")
scaling = _scaling()
st.plotly_chart(V.build_scaling(scaling), width="stretch", key="scaling")
kind, text = ev.verdict_scaling(scaling)
getattr(st, kind)(text)
st.dataframe(
    pd.DataFrame([
        {"Lieferzeit Händler/Lager": f"{r.Lr}/{r.Lw}", "S1": r.S1, "S2": r.S2, "Zustände der gemeinsamen Rechnung": r.dp_states, "Rechenschritte der Zerlegung": r.decomposition_steps,
         "Gemeinsame DP": "gerechnet" if r.dp_solved else "nur gezählt", "Zeit der DP (Größenordnung)": r.dp_time}
        for r in scaling
    ]),
    hide_index=True, width="stretch",
    column_config={"Zustände der gemeinsamen Rechnung": st.column_config.NumberColumn(format="%d"), "Rechenschritte der Zerlegung": st.column_config.NumberColumn(format="%d")},
)
st.caption(
    "Raster: Händlerbestand von −20 bis S1 + 7, Lagerbestand 0 bis S2 − S1 + 16, je Pipelineplatz 0 bis 13. Zustände und Rechenschritte sind verschiedene Einheiten: die gemeinsame DP braucht je Iteration Zustände mal Aktionen Schritte. "
    "Die Zeiten sind Größenordnungen aus einer Messung unter Last und keine feste Zahl; „nur gezählt“ heißt: die gemeinsame Rechnung wurde dort nicht ausgeführt."
)

# --- Handbeispiel ---------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("📝 Handbeispiel von Hand")
ex = _example()
dec_ex = ex["decomposition"]
st.markdown(
    "Lieferzeit 1/1, Nachfrage D = 0, 1, 2 mit Wahrscheinlichkeit 1/4, 1/2, 1/4 (Mittel 1), Lagerkostensatz des Händlers 1, des Lagers 0,5, Fehlmengenkosten p = 3, Bestellgrenze 4. "
    "Echelon-Sätze: Händlerstufe 1 − 0,5 = 0,5 je Stück Händlerbestand und 3 + 0,5 = 3,5 je fehlendes Stück; Lagerstufe 0,5 je Stück Echelon-Bestand. Dieses Beispiel ist fest und hängt nicht von den Reglern ab."
)
st.markdown(
    f"1. **Nachfrage über 2 Perioden** (Lieferzeit des Händlers plus eins): (1, 4, 6, 4, 1)/16 für 0 bis 4. Code: {', '.join(str(Fraction(v).limit_denominator(64)) for v in ex['demand2'])}.\n"
    f"2. **Händlerstufe** L1(y) für y = 0 bis 5: {', '.join(de(v, 2) for v in ex['L1'])}. Das Minimum liegt bei **S1 = {dec_ex.S1}** (Quantil zum Verhältnis 3,5/4 = 0,875: kleinstes y mit F(y) ≥ 0,875 ist 3).\n"
    f"3. **Lagerstufe** G2(y; S1 = 3) für y = 2 bis 6: {', '.join(de(v, 4) for v in ex['G2'])}. Das Minimum liegt bei **S2 = {dec_ex.S2}** mit **g = {de(dec_ex.g, 4)}**."
)
st.markdown("4. **Kette der Politik (3, 4):** sechs wiederkehrende Zustände (x, w) mit Wahrscheinlichkeit und Kosten der Periode:")
st.dataframe(
    pd.DataFrame([
        {"Zustand (x, w)": f"({s['x']}, {s['w']})", "Wahrscheinlichkeit": str(s["pi"]), "Bestellung (qr, qw)": f"({s['qr']}, {s['qw']})", "Kosten der Periode": s["cost"]}
        for s in ex["chain"]
    ]),
    hide_index=True, width="stretch", column_config={"Kosten der Periode": st.column_config.NumberColumn(format="%.1f")},
)
g_frac = sum(s["pi"] * Fraction(s["cost"]) for s in ex["chain"])
st.markdown(f"Mittel: (4·1 + 1,5·3 + 2·2 + 1,5·6 + 2·1 + 2,5·3)/16 = **{g_frac}** = {de(float(g_frac), 4)}, gleich dem Minimum von G2 aus der Zerlegung.")
st.markdown("5. **Drei Sichten im Zustand (x, w) = (0, 2)** bei S1 = 3 und Sw = 3: die Anforderung des Händlers ist 3, das Lager hat 2.")
orders = ex["orders"]
st.dataframe(
    pd.DataFrame([{"Sicht": C.VIEW_LABEL[k], "Bestellung des Lagers qw": orders[k][1], "Rechnung": {"A": "3 − 2 = 1", "B": "3 − (2 − 2) = 3", "C": "3 − (2 − 3) = 4"}[k]} for k in C.VIEWS]),
    hide_index=True, width="stretch",
)
loc = ex["local"]
st.caption(
    f"Gemeinsame DP des Beispiels (91 Zustände): {de(loc['dp'], 4)}. Beste Niveaus der lokalen Regeln durch Aufzählung: "
    + "; ".join(f"Sicht {k}: ({loc[k]['S1']}, {loc[k]['Sw']}), {pct(loc[k]['gap'], 2, sign=True)}" for k in C.VIEWS) + "."
)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        f"""
| Annahme | Was passiert, wenn sie verletzt ist | Stand |
|---|---|---|
| **Seriell, Rückstand nur beim Händler, deterministische Lieferzeiten, stationäre unabhängige Nachfrage, keine Fixkosten, keine Kapazitätsgrenzen, zentrale Steuerung** (Voraussetzungen der Sätze nach van Houtum 2006 und Chao/Zhou 2006) | Die Optimalität der Zerlegung ist ohne diese Voraussetzungen nicht gesichert. | Nur das Modell der Demo gerechnet |
| **Poisson-Nachfrage, Mittel 3, Lagerkostensatz des Händlers 1** | Andere Verteilungen und andere Mittel sind nicht gemessen. | Nicht gemessen |
| **Bestellgrenze {C.Q_MAX} je Stufe** | In der Hauptzelle liegt die Politik in {de(R.get_cell(1, 1, 0.4, 9)['cap_dp'] * 100, 4)} % der Zeit an der Grenze, in allen 48 Zellen höchstens {de(tally['max_cap'] * 100, 4)} %. Die Gleichheit Zerlegung = gemeinsame DP wurde in der Hauptzelle auch für die Grenzen 4, 5, 6, 7, 8, 10 und 13 gemessen (mit neu abgestimmten Niveaus); die Niveaus der Zerlegung selbst verlieren bei Grenze 4 / 5 / 6 +7,71 / +1,28 / +0,05 %. Grenze 3 ist unbrauchbar. | Messbefund, nicht bewiesen |
| **Lokale Niveaus per Bergsteigen** | Die Niveaus der lokalen Regeln A und B sind aus drei Startpunkten gesucht, in Lieferzeit 1/1 gegen das volle Fenster geprüft (24 von 24 Vergleichen gleich: 12 Zellen, je Regel A und B); kein Beweis für das globale Optimum. | Heuristik |
| **Zeitschema** | Bestellung zu Beginn der Periode, Ankunft bei Lieferzeit 1 am Ende der laufenden Periode nach der Nachfrage; Kosten auf dem Bestand zu Beginn der Periode. Die Formeln L1 und G2 sind eigene Herleitungen für dieses Schema. | Modellfestlegung |
| **Rückstand auch beim Lager, stochastische Lieferzeiten, mehr als zwei Stufen, unkoordinierte Wahl der Niveaus** | Nicht gerechnet; die Aussagen dieser Demo gelten dafür nicht. | Offen |
| **Randfall h_w = Händlersatz** | Der Echelon-Satz der Händlerstufe wäre 0 und S1 nicht eindeutig. Der Regler endet bei 0,8. | Nicht gerechnet |
| **Quellenlage** | Clark/Scarf 1960 ist nur nach dem Abstract bekannt; die Struktur der Zerlegung stützt sich auf van Houtum (2006) und Chao/Zhou (2006). Keine gelesene Quelle nennt eine Zahl für den Verlust einer lokalen Regel: alle Lücken sind eigene Messungen. | Gelesen / nur Abstract |

**Zeiten:** Rechenzeiten sind Größenordnungen unter Last, keine festen Zahlen. **Streamlit Cloud:** das Raster der gemeinsamen DP bei Lieferzeit 2/2 braucht rund 200 MB Arbeitsspeicher.
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben: Nachfrage $D$ je Periode (Poisson, Mittel 3), Lieferzeiten $L_r$ (Lager → Händler) und $L_w$ (Lieferant → Lager), Kostensätze $h_r = 1$ (Händlerbestand), $h_w$ (Lager und Weg zum Händler), $p$ (Fehlmenge), Bestellkosten 0.
Gesucht: Bestellmengen $q_r, q_w$ je Periode mit minimalen mittleren Kosten je Periode.

| Symbol | Bedeutung |
|---|---|
| $x$ | Händler-Nettobestand am Periodenanfang (negativ: Rückstand) |
| $w$ | Lagerbestand am Periodenanfang |
| $q_r$, $q_w$ | Bestellung des Händlers beim Lager, des Lagers beim Lieferanten (höchstens $Q = 13$) |
| $y$ | Echelon-Bestand nach der Bestellung |
| $S_1$, $S_2$ | Niveaus der Händler- und der Lagerstufe |
| $L_1$, $G_2$ | Kosten je Periode der Händler- und der Lagerstufe |
| $D^{(n)}$ | Nachfrage über $n$ Perioden (Faltung) |
| $S_w$ | Niveau des Lagers bei der lokalen Regel |

**Kosten je Periode** im Zustand $(x, w)$ mit $\Sigma r$ unterwegs zum Händler:
$$c(x,w) = h_w\,(w + \textstyle\sum r) + \mathbb{E}\big[h_r\,(x-D)^+ + p\,(D-x)^+\big].$$

**Gemeinsame Rechnung** (Bellman, mittlere Kosten $g$ je Periode, relative Wertefunktion $h$):
$$g + h(s) = \min_{q_r \le w,\; q_w}\Big\{\, c(s) + \mathbb{E}\, h(s') \Big\}.$$

**Schritt 1, Händlerstufe** (Echelon-Sätze $h_r - h_w$ und $p + h_w$, Nachfrage über $L_r + 1$ Perioden):
$$L_1(y) = \mathbb{E}\Big[(h_r - h_w)\,\big(y - D^{(L_r+1)}\big)^+ + (p + h_w)\,\big(D^{(L_r+1)} - y\big)^+\Big], \qquad S_1 = \arg\min_y L_1(y).$$

**Schritt 2, Lagerstufe** (induzierte Kostenfunktion, Nachfrage über $L_w$ Perioden):
$$G_2(y) = h_w\big(y - (L_w+1)\,\mathbb{E}D\big) + \mathbb{E}\, L_1\big(\min\{S_1,\; y - D^{(L_w)}\}\big), \qquad S_2 = \arg\min_y G_2(y), \quad g = G_2(S_2).$$

**Echelon-Politik:** der Händler bestellt $q_r = \min\{(S_1 - x - \textstyle\sum r)^+,\; w,\; Q\}$, das Lager bestellt
$$q_w = \min\{(S_2 - w - \textstyle\sum r - x - \sum u)^+,\; Q\}.$$

**Lokale Regel** (Niveau $S_w$; $a = (S_1 - x - \sum r)^+$ ist die Anforderung des Händlers):
$$q_w^{A} = (S_w - w - \textstyle\sum u)^+, \qquad q_w^{B} = (S_w - w + q_r - \sum u)^+, \qquad q_w^{C} = (S_w - w + a - \sum u)^+.$$

Mit $S_w = S_2 - S_1$ bestellt Sicht C auf den erreichten Zuständen dasselbe wie die Echelon-Politik (eigene Herleitung und Messbefund in 48 von 48 Zellen, keine gelesene Quelle).

**Literatur.** G. J. van Houtum (2006): Multiechelon Production/Inventory Systems: Optimal Policies, Heuristics, and Algorithms (INFORMS TutORials; gelesen wurde die Arbeitsfassung der TU Eindhoven). X. Chao, S. X. Zhou (2006):
Newsvendor Bounds and Heuristics for Optimal Policy of Serial Supply Chains (ISORA 2006, mit dem Zusatz „with and without Expedited Shippings“; frei lesbare Fassung, nicht im Original gelesen). A. J. Clark, H. Scarf (1960), Management Science 6(4), 475–490: nur nach dem Abstract.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Bestandspolitiken: von Base-Stock bis Regelwahl](https://sebastianhanisch.net/konzepte-bestandspolitiken.html)."
)
