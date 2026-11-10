"""Auswertung: Einstellungen, Analyse einer Zelle, Verlauf über denselben Nachfragestrom und die Verdikte der App (reine Funktionen).

Jedes Verdikt ist eine reine Funktion `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`. Zahlen im Text stehen mit Dezimalkomma."""

from dataclasses import dataclass

import lah_constants as C
import lah_decompose as D
import lah_results as R
import lah_simulate as S
from lah_model import Params

EQUAL_TOL = 1e-5        # Kosten gelten als gleich, wenn sie sich um weniger als dies unterscheiden (fünf gedruckte Stellen)
SMALL_GAP = 0.005       # Lücken unter 0,005 % gelten als 0,00 %


def de(x: float, digits: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma (und auf Wunsch Vorzeichen)."""
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace(".", ",")


def pct(x: float, digits: int = 1, sign: bool = False) -> str:
    return de(x, digits, sign) + " %"


@dataclass(frozen=True, kw_only=True)
class Settings:
    lr: int = C.DEFAULT_LR
    lw: int = C.DEFAULT_LW
    hw: float = C.DEFAULT_HW
    p: int = C.DEFAULT_P
    view: str = C.DEFAULT_VIEW
    d1: int = 0
    d2: int = 0
    seed: int = C.DEFAULT_SEED
    show: int = C.DEFAULT_SHOW


@dataclass(frozen=True)
class Analysis:
    cell: dict
    dec: D.Decomposition     # Zerlegung live gerechnet (Millisekunden)
    S1: int                  # Echelon-Niveaus der Politik, die die App zeigt: Zerlegung plus Abweichung der Regler
    S2: int
    g_levels: float          # geschlossene Formel für (S1, S2)
    g_dp: float
    gap_levels: float        # Lücke dieser Politik zur DP in Prozent (Formel gegen DP mit Bestellgrenze: höchstens 1e-5 Unterschied)
    gaps: dict               # Lücke je Sicht A, B, C in Prozent


def params_of(settings: Settings) -> Params:
    return Params(Lr=settings.lr, Lw=settings.lw, h_w=settings.hw, p=float(settings.p), Q=C.Q_MAX)


def analyse(settings: Settings) -> Analysis:
    cell = R.get_cell(settings.lr, settings.lw, settings.hw, settings.p)
    params = params_of(settings)
    dec = D.solve(params)
    S1, S2 = dec.S1 + settings.d1, dec.S2 + settings.d2
    g_levels = D.echelon_cost(params, S1, S2)
    return Analysis(
        cell=cell, dec=dec, S1=S1, S2=S2, g_levels=g_levels, g_dp=cell["g_dp"], gap_levels=R.gap_percent(g_levels, cell["g_dp"]),
        gaps={k: R.rule_gap(cell, k) for k in C.VIEWS},
    )


@dataclass(frozen=True)
class Run:
    demands: list
    echelon: S.Trace
    local: S.Trace
    lag_echelon: dict       # Lag k -> Anteil der Perioden, in denen die Lagerbestellung der Nachfrage von Periode t - k entspricht (lange Läufe)
    lag_local: dict
    mean_echelon: float
    mean_local: float
    shown: slice


def run(settings: Settings, length: int = C.RUN_LENGTH, burn: int = C.BURN_IN) -> Run:
    """Echelon-Politik und gewählte lokale Regel über denselben Nachfragestrom (Zufalls-Seed der Regler)."""
    params = params_of(settings)
    cell = R.get_cell(settings.lr, settings.lw, settings.hw, settings.p)
    dec = D.solve(params)
    S1, S2 = dec.S1 + settings.d1, dec.S2 + settings.d2
    rule = cell["rules"][settings.view]
    demands = S.demand_stream(params.pmf(), burn + length, settings.seed)
    ech = S.simulate(params, "echelon", S1, S2, demands, burn=burn)
    loc = S.simulate(params, settings.view, rule["S1"], rule["S2"], demands, burn=burn)
    lags = (1, 2, 3)
    return Run(
        demands=demands, echelon=ech, local=loc,
        lag_echelon={k: S.lag_share(ech.qw, demands, k, burn) for k in lags},
        lag_local={k: S.lag_share(loc.qw, demands, k, burn) for k in lags},
        mean_echelon=ech.mean_cost, mean_local=loc.mean_cost, shown=slice(burn, burn + settings.show),
    )


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------

def verdict_decomposition(g_dp: float, g_ech: float, S1: int, S2: int, n_equal: int, n_cells: int) -> tuple[str, str]:
    """Kernaussage: trifft die Echelon-Politik mit den Niveaus der Zerlegung die gemeinsame DP?"""
    head = f"Echelon-Politik mit den Niveaus der Zerlegung (S1 = {S1}, S2 = {S2}): {de(g_ech, 5)} je Periode; gemeinsame DP: {de(g_dp, 5)}."
    if abs(g_ech - g_dp) < EQUAL_TOL:
        return "success", head + f" Kein Unterschied auf fünf Stellen. Dasselbe gilt in {n_equal} von {n_cells} gemessenen Zellen."
    return "warning", head + f" Unterschied {de(g_ech - g_dp, 5, sign=True)}; über alle gemessenen Zellen gilt Gleichheit in {n_equal} von {n_cells}."


def verdict_view(view: str, gap: float, delay_equal: int, delay_total: int) -> tuple[str, str]:
    """Lücke der lokalen Regel in der gewählten Sicht des Lagers (Prozent der DP-Kosten)."""
    if view == "C":
        if gap < SMALL_GAP:
            return "success", "Sicht C: Das Lager sieht die volle Bestellung des Händlers und bekommt sein Niveau aus der Zerlegung (Sw = S2 - S1). Die Regel verliert nichts gegenüber der gemeinsamen DP (0,00 %)."
        return "warning", f"Sicht C kostet {pct(gap, 2)} mehr als die gemeinsame DP."
    if gap < SMALL_GAP:
        return "info", f"Sicht {view}: kein Unterschied zur gemeinsamen DP in dieser Zelle."
    if view == "A":
        cause = (f"Das Lager sieht die Bestellung des Händlers erst eine Periode später; das entspricht einer um eine Periode längeren Lieferzeit des Lagers "
                 f"(Deutung, gestützt darauf, dass A bei Lagerlieferzeit 1 so viel kostet wie B bei 2: {delay_equal} von {delay_total} Paaren).")
    else:
        cause = "Das Lager sieht die Lieferung an den Händler, aber nicht den Teil der Bestellung, den es nicht liefern konnte."
    return "warning", f"Lokale Regel, Sicht {view}: **{pct(gap, 2, sign=True)}** gegenüber der gemeinsamen DP (beste Niveaus der Regel). {cause}"


def verdict_levels(d1: int, d2: int, S1: int, S2: int, g_levels: float, g_zerlegung: float) -> tuple[str, str]:
    """Wirkung der Abweichung der Echelon-Niveaus von der Zerlegung (geschlossene Formel, ohne Bestellgrenze)."""
    if d1 == 0 and d2 == 0:
        return "info", f"Niveaus der Zerlegung: S1 = {S1}, S2 = {S2}."
    extra = 100.0 * (g_levels / g_zerlegung - 1.0)
    return "info", f"Mit S1 = {S1} und S2 = {S2} kostet die Echelon-Politik {de(g_levels, 5)}: {pct(extra, 2, sign=True)} gegenüber den Niveaus der Zerlegung (geschlossene Formel, ohne Bestellgrenze)."


def verdict_dp_live(g_live: float, g_stored: float, seconds: float, states: int) -> tuple[str, str]:
    diff = abs(g_live - g_stored)
    n_states = f"{states:,}".replace(",", ".")
    text = f"Gemeinsame DP eben gerechnet: {de(g_live, 5)} je Periode auf {n_states} Zuständen ({de(seconds, 1)} s auf diesem Rechner)"
    if diff < 1e-6:
        return "success", text + f"; gleich dem vorgerechneten Wert ({de(g_stored, 5)}, Unterschied {diff:.0e})."
    return "warning", text + f"; weicht vom vorgerechneten Wert ({de(g_stored, 5)}) um {diff:.1e} ab."


def verdict_trend(kind: str, hw_ok: int, hw_n: int, p_ok: int, p_n: int, p_exceptions: list) -> tuple[str, str]:
    """Wovon hängt die Lücke der lokalen Regel ab: Zählungen über die Zellen."""
    exc = ""
    if p_exceptions:
        names = "; ".join(f"Lieferzeit {lr}/{lw}, h_w {de(h, 1)}" for lr, lw, h in p_exceptions)
        exc = f" Ausnahme: {names}."
    return "info", (f"Sicht {kind}: Die Lücke wächst mit h_w in {hw_ok} von {hw_n} Reihen (feste Lieferzeiten und p) "
                    f"und fällt mit p in {p_ok} von {p_n} Reihen (feste Lieferzeiten und h_w).{exc}")


def verdict_delay(pairs: list) -> tuple[str, str]:
    """A bei Lagerlieferzeit 1 gegen B bei Lagerlieferzeit 2 (beste Niveaus je Regel)."""
    n = len(pairs)
    equal = sum(1 for q in pairs if q["diff"] < EQUAL_TOL)
    worst = max(q["diff"] for q in pairs)
    kind = "success" if equal == n else "warning"
    return kind, (f"Die lokale Regel A bei Lagerlieferzeit 1 kostet so viel wie die Regel B bei Lagerlieferzeit 2: gleich auf fünf Stellen in {equal} von {n} Paaren "
                  f"(größter Unterschied {worst:.1e}). Ein Mehrwissen des Lagers um eine Periode wirkt wie eine um eine Periode kürzere Lieferzeit "
                  f"(gemessen für Lagerlieferzeit 1 gegen 2 in {n} Zellen dieses Modells; Deutung).")


def verdict_scaling(rows: list) -> tuple[str, str]:
    """Zustandszahl der gemeinsamen Rechnung gegen den Aufwand der Zerlegung (Tafel Lieferzeit 1/1 bis 4/4)."""
    small = next(r for r in rows if (r.Lr, r.Lw) == (1, 1))
    big = next(r for r in rows if (r.Lr, r.Lw) == (4, 4))
    fmt = lambda n: f"{n:,}".replace(",", ".")  # noqa: E731
    return "info", (f"Gemeinsame Rechnung: {fmt(small.dp_states)} Zustände bei Lieferzeit 1/1, {fmt(big.dp_states)} bei 4/4 (nur gezählt, die Indextafel allein ginge in den Terabyte-Bereich). "
                    f"Zerlegung: {fmt(small.decomposition_steps)} Rechenschritte bei 1/1, {fmt(big.decomposition_steps)} bei 4/4.")


def verdict_lag(local_kind: str, lag_echelon: dict, lag_local: dict) -> tuple[str, str]:
    """Bestellt das Lager die Nachfrage der letzten Periode (Lag 1) oder einer früheren? Anteile über den langen Lauf."""
    best_e = max(lag_echelon, key=lag_echelon.get)
    best_l = max(lag_local, key=lag_local.get)
    return "info", (f"Über den langen Lauf bestellt die Echelon-Politik in {pct(100 * lag_echelon[best_e], 1)} der Perioden genau die Nachfrage von Periode t - {best_e}; "
                    f"die lokale Regel (Sicht {local_kind}) in {pct(100 * lag_local[best_l], 1)} der Perioden die von t - {best_l}.")


def solve_dp_live(cell: dict) -> tuple:
    """Gemeinsame DP der Zelle jetzt rechnen (Raster aus der Zelle): (g, Zustände, Sekunden, Iterationen)."""
    import time

    import lah_dp
    from lah_model import StateSpace

    r = cell["raster"]
    params = Params(Lr=cell["Lr"], Lw=cell["Lw"], h_w=cell["h_w"], p=cell["p"], Q=C.Q_MAX, xmin=r["xmin"], xmax=r["xmax"], W=r["W"])
    t0 = time.perf_counter()
    space = StateSpace(params)
    res = lah_dp.value_iteration(space)
    return res.g, space.n_states, time.perf_counter() - t0, res.iterations
