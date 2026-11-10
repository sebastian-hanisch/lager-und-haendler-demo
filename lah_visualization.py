"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lah_constants as C
import lah_decompose as D
import lah_results as R
from lah_model import Params


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_rule_costs(cell: dict):
    """Gestapelte Kosten je Periode der drei Sichten und der Echelon-Politik (= gemeinsame DP), nach Anteilen getrennt."""
    names = {"echelon": "Echelon = DP", "A": "lokal A", "B": "lokal B", "C": "lokal C"}
    parts = (("hold_w", "Lager und Weg zum Händler halten", C.COLOR_HOLD_W), ("hold_r", "Händler halten", C.COLOR_HOLD_R), ("short", "Rückstand (Fehlmengen)", C.COLOR_SHORT))
    fig = go.Figure()
    labels = [names[k] for k in names]
    for key, title, color in parts:
        fig.add_trace(go.Bar(x=labels, y=[cell["rules"][k][key] for k in names], name=title, marker_color=color,
                             hovertemplate="%{x}: %{y:.3f}<extra>" + title + "</extra>"))
    totals = [cell["rules"][k]["g"] for k in names]
    fig.add_trace(go.Scatter(
        x=labels, y=totals, mode="text", showlegend=False, textposition="top center", hoverinfo="skip",
        text=[f"{_de(g, 3)}" + ("" if k == "echelon" else f" ({_de(R.rule_gap(cell, k), 2)} % über DP)".replace("(", "(+")) for k, g in zip(names, totals)],
    ))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Kosten je Periode", range=[0, max(totals) * 1.15])
    return _base(fig, height=380)


def build_decomposition(params: Params, dec: D.Decomposition, S1_shown: int, S2_shown: int):
    """Die zwei eindimensionalen Rechnungen: L1(y) mit Minimum S1, G2(y; S1) mit Minimum S2; die gezeigten Niveaus als Markierung."""
    ys1 = np.arange(max(dec.S1 - 10, 0), dec.S1 + 11)
    ys2 = np.arange(max(dec.S2 - 10, 0), dec.S2 + 11)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Schritt 1: Händlerstufe L1(y)", f"Schritt 2: Lagerstufe G2(y), mit S1 = {dec.S1}"))
    fig.add_trace(go.Scatter(x=ys1, y=D.retailer_cost(params, ys1), mode="lines+markers", line=dict(color=C.COLOR_ECHELON, width=3), name="L1(y)",
                             hovertemplate="y = %{x}: %{y:.4f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ys2, y=D.warehouse_cost(params, ys2, dec.S1), mode="lines+markers", line=dict(color=C.COLOR_A, width=3), name="G2(y)",
                             hovertemplate="y = %{x}: %{y:.4f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[dec.S1], y=[float(D.retailer_cost(params, np.array(dec.S1)))], mode="markers", marker=dict(size=14, symbol="star", color="black"),
                             name="Minimum", hovertemplate="S1 = %{x}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[dec.S2], y=[dec.g], mode="markers", marker=dict(size=14, symbol="star", color="black"), showlegend=False,
                             hovertemplate="S2 = %{x}<extra></extra>"), row=1, col=2)
    if (S1_shown, S2_shown) != (dec.S1, dec.S2):
        fig.add_trace(go.Scatter(x=[S1_shown], y=[float(D.retailer_cost(params, np.array(S1_shown)))], mode="markers", marker=dict(size=11, symbol="x", color="gray"),
                                 name="gewähltes Niveau", hovertemplate="S1 = %{x}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[S2_shown], y=[float(D.warehouse_cost(params, np.array([S2_shown]), dec.S1)[0])], mode="markers",
                                 marker=dict(size=11, symbol="x", color="gray"), showlegend=False, hovertemplate="S2 = %{x}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Echelon-Bestand y nach der Bestellung", row=1, col=1)
    fig.update_xaxes(title_text="Echelon-Bestand y nach der Bestellung", row=1, col=2)
    fig.update_yaxes(title_text="Kosten je Periode", row=1, col=1)
    return _base(fig, height=360)


def build_trace(run, kind: str, show: int):
    """Verlauf über denselben Nachfragestrom: Nachfrage, Händlerbestand, Lagerbestand, Bestellung des Lagers (Echelon gegen die gewählte lokale Regel)."""
    sl = run.shown
    t = list(range(show))
    ech, loc = run.echelon, run.local
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.18, 0.27, 0.27, 0.28],
                        subplot_titles=("Nachfrage je Periode", "Händlerbestand am Periodenanfang (negativ = Rückstand)", "Lagerbestand am Periodenanfang", "Bestellung des Lagers beim Lieferanten"))
    fig.add_trace(go.Bar(x=t, y=run.demands[sl], marker_color=C.COLOR_DEMAND, name="Nachfrage", hovertemplate="t = %{x}: %{y}<extra>Nachfrage</extra>"), row=1, col=1)
    color_local = C.VIEW_COLOR[kind]
    for row, key in ((2, "x"), (3, "w"), (4, "qw")):
        fig.add_trace(go.Scatter(x=t, y=getattr(ech, key)[sl], mode="lines+markers", line=dict(color=C.COLOR_ECHELON, width=2), name="Echelon-Politik", showlegend=(row == 2),
                                 hovertemplate="t = %{x}: %{y}<extra>Echelon</extra>"), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=getattr(loc, key)[sl], mode="lines+markers", line=dict(color=color_local, width=2, dash="dot"), name=f"lokal, Sicht {kind}", showlegend=(row == 2),
                                 hovertemplate="t = %{x}: %{y}<extra>lokal " + kind + "</extra>"), row=row, col=1)
    fig.update_xaxes(title_text="Periode nach der Einschwingzeit", row=4, col=1)
    return _base(fig, height=720, legend_y=-0.08)


def build_gap_lines(rows: list, kind: str, lr: int, lw: int, current_hw: float, current_p: int):
    """Lücke der lokalen Regel zur DP über h_w, je eine Linie für p = 4, 9, 19 (feste Lieferzeiten); die gewählte Zelle ist markiert."""
    fig = go.Figure()
    palette = {4: "#56B4E9", 9: "#0072B2", 19: "#000000"}
    for p in C.P_GRID:
        sel = sorted((r for r in rows if r["Lr"] == lr and r["Lw"] == lw and r["p"] == p), key=lambda r: r["h_w"])
        fig.add_trace(go.Scatter(x=[r["h_w"] for r in sel], y=[r["gap_" + kind] for r in sel], mode="lines+markers", name=f"p = {p}", line=dict(color=palette[p], width=3),
                                 hovertemplate="h_w = %{x}: %{y:.2f} %<extra>p = " + str(p) + "</extra>"))
    cur = next(r for r in rows if (r["Lr"], r["Lw"], r["h_w"], r["p"]) == (lr, lw, current_hw, current_p))
    fig.add_trace(go.Scatter(x=[current_hw], y=[cur["gap_" + kind]], mode="markers", marker=dict(size=16, symbol="circle-open", line=dict(width=3, color="#D55E00")),
                             name="gewählte Zelle", hovertemplate="gewählte Zelle<extra></extra>"))
    fig.update_xaxes(title_text="Lagerkostensatz h_w", tickvals=list(C.H_W_GRID))
    fig.update_yaxes(title_text=f"Lücke der Sicht {kind} zur DP (%)", rangemode="tozero")
    return _base(fig, height=340)


def build_delay(pairs: list):
    """Kosten von A bei Lagerlieferzeit 1 (x) gegen B bei Lagerlieferzeit 2 (y), je Zelle ein Punkt; gleiche Kosten liegen auf der Diagonalen."""
    lo = min(min(q["g_A"], q["g_B_next"]) for q in pairs) * 0.97
    hi = max(max(q["g_A"], q["g_B_next"]) for q in pairs) * 1.03
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="gray", dash="dash"), name="gleiche Kosten", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[q["g_A"] for q in pairs], y=[q["g_B_next"] for q in pairs], mode="markers", marker=dict(size=10, color=C.COLOR_A), name="Zelle",
                             text=[f"Lr {q['Lr']}, h_w {q['h_w']}, p {q['p']}" for q in pairs], hovertemplate="%{text}<br>A: %{x:.5f}<br>B: %{y:.5f}<extra></extra>"))
    fig.update_xaxes(title_text="Regel A bei Lagerlieferzeit 1 (Kosten je Periode)", range=[lo, hi])
    fig.update_yaxes(title_text="Regel B bei Lagerlieferzeit 2", range=[lo, hi], scaleanchor="x", scaleratio=1)
    return _base(fig, height=380)


def build_scaling(rows: list):
    """Zustandszahl der gemeinsamen Rechnung gegen Rechenschritte der Zerlegung über die Lieferzeiten (logarithmische Achse)."""
    labels = [f"{r.Lr}/{r.Lw}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r.dp_states for r in rows], name="Zustände der gemeinsamen Rechnung", marker_color=[C.COLOR_A if r.dp_solved else "rgba(213,94,0,0.35)" for r in rows],
                         hovertemplate="%{x}: %{y:,.0f} Zustände<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=[r.decomposition_steps for r in rows], mode="lines+markers", name="Rechenschritte der Zerlegung", line=dict(color=C.COLOR_ECHELON, width=3),
                             hovertemplate="%{x}: %{y:,.0f} Schritte<extra></extra>"))
    fig.update_yaxes(type="log", title_text="Anzahl (logarithmisch)")
    fig.update_xaxes(title_text="Lieferzeit Händler/Lager (kräftig = gerechnet, blass = nur gezählt)")
    return _base(fig, height=380)
