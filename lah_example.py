"""Handbeispiel: Lieferzeit 1/1, Nachfrage 0, 1, 2 mit Wahrscheinlichkeit 1/4, 1/2, 1/4, h_Händler = 1, h_Lager = 0,5, p = 3, Bestellgrenze 4 je Stufe.

Die Zerlegung, die Kette der Politik (3, 4) und das Ergebnis lassen sich mit Bleistift rechnen (Plan und README); dieses Modul rechnet dieselben Größen mit dem Code
der Demo, damit die App Handrechnung und Code nebeneinander zeigt. Das Beispiel ist fest und hängt nicht von den Reglern ab."""

from fractions import Fraction

import numpy as np

import lah_chain as CH
import lah_constants as C
import lah_decompose as D
import lah_demand as demand
import lah_dp
import lah_policies as P
from lah_model import Params, StateSpace

EXAMPLE_STATE = (0, 2)       # Zustand (x, w) für den Vergleich der drei Sichten
EXAMPLE_SW = 3               # Niveau des Lagers für die drei Sichten in diesem Zustand
EXAMPLE_S1, EXAMPLE_S2 = 3, 4


def example_params() -> Params:
    return Params(**C.EXAMPLE)


def retailer_values(params: Params, ys=range(0, 6)) -> list:
    return [float(D.retailer_cost(params, np.array(y))) for y in ys]


def warehouse_values(params: Params, S1: int = EXAMPLE_S1, ys=range(2, 7)) -> list:
    return [float(D.warehouse_cost(params, np.array([y]), S1)[0]) for y in ys]


def two_period_demand(params: Params) -> list:
    return [float(v) for v in demand.convolve_n(params.pmf(), params.Lr + 1)]


def chain_states(params: Params, S1: int = EXAMPLE_S1, S2: int = EXAMPLE_S2) -> list:
    """Wiederkehrende Zustände der Kette der Echelon-Politik: (x, w), Wahrscheinlichkeit als Bruch, Aktion (qr, qw) und Kosten der Periode."""
    space = StateSpace(params)
    pol = P.policy_actions(space, "echelon", S1, S2)
    res = CH.evaluate(space, pol, method="direct")
    out = []
    for s in np.flatnonzero(res.pi > 1e-12):
        qr, qw = divmod(int(pol[s]), space.nq)
        out.append(dict(x=int(space.x[s]), w=int(space.w[s]), pi=Fraction(float(res.pi[s])).limit_denominator(2000), qr=qr, qw=qw, cost=float(space.cost[s])))
    return out


def local_best(params: Params) -> dict:
    """Beste Niveaus (S1, Sw) der drei Sichten durch Aufzählung (91 Zustände); Kosten und Lücke zur DP."""
    space = StateSpace(params)
    dp = lah_dp.value_iteration(space, tol=1e-12)
    out = {}
    for kind in C.VIEWS:
        best = None
        for s1 in range(1, 6):
            for sw in range(0, 9):
                g = CH.evaluate(space, P.policy_actions(space, kind, s1, sw), method="direct").g
                if best is None or g < best[0] - 1e-12:
                    best = (g, s1, sw)
        out[kind] = dict(g=best[0], S1=best[1], Sw=best[2], gap=100.0 * (best[0] / dp.g - 1.0))
    out["dp"] = dp.g
    return out


def orders_in_state(params: Params, S1: int = EXAMPLE_S1, Sw: int = EXAMPLE_SW, state: tuple = EXAMPLE_STATE) -> dict:
    """Bestellungen (qr, qw) der drei Sichten im Zustand (x, w) bei gleichen Niveaus."""
    x, w = state
    return {k: tuple(int(v) for v in P.orders(params, k, S1, Sw, x, w, 0, 0)) for k in C.VIEWS}


def report() -> dict:
    params = example_params()
    dec = D.solve(params, lo=-10, hi=20)
    states = chain_states(params)
    return dict(
        params=params, demand2=two_period_demand(params), L1=retailer_values(params), G2=warehouse_values(params), decomposition=dec,
        chain=states, g_chain=float(sum(s["pi"] * Fraction(s["cost"]).limit_denominator(1000) for s in states)),
        local=local_best(params), orders=orders_in_state(params),
    )
