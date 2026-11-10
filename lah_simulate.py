"""Simulation (ganzzahliger Erzeuger, unabhängig von der numpy-Version): Nachfragestrom und Verlauf einer Regel (nur numpy und Standardbibliothek).

Python-hash() ist kein Zufalls-Seed (pro Prozess gesalzen) und numpy-Ströme sind zwischen Versionen nicht stabil; deshalb SplitMix64 mit Inversion der
Verteilungsfunktion. Gleicher Zufalls-Seed, gleiche Verteilung, gleiche Länge: gleicher Nachfragestrom auf jeder Plattform."""

from dataclasses import dataclass

import numpy as np

from lah_model import Params
from lah_policies import orders

MASK = 0xFFFFFFFFFFFFFFFF


class SplitMix:
    """SplitMix64: gleichverteilte Zahlen in [0, 1) aus einem ganzzahligen Zufalls-Seed."""

    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_uniform(self) -> float:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z = z ^ (z >> 31)
        return (z >> 11) / float(1 << 53)


def demand_stream(pmf, n: int, seed: int) -> list[int]:
    """n ganzzahlige Nachfragen aus der Verteilung pmf (Inversion der Verteilungsfunktion)."""
    rng = SplitMix(seed)
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0
    u = np.array([rng.next_uniform() for _ in range(n)])
    return np.searchsorted(cdf, u, side="right").tolist()


@dataclass(frozen=True)
class Trace:
    x: list     # Händlerbestand am Periodenanfang
    w: list     # Lagerbestand am Periodenanfang
    qr: list    # Bestellung des Händlers beim Lager
    qw: list    # Bestellung des Lagers beim Lieferanten
    cost: list  # Kosten der Periode (mit der realisierten Nachfrage)
    mean_cost: float   # Mittel über alle Perioden ab `burn`


def simulate(params: Params, kind: str, S1: int, S2: int, demands, burn: int = 0) -> Trace:
    """Verlauf einer Regel über einen vorgegebenen Nachfragestrom; Anfangszustand x = 3, w = 4, Pipelines leer.

    Das Mittel der Kosten zählt erst ab Periode `burn` (Einschwingzeit: ein leerer Start verfälscht sonst das Mittel)."""
    Lr, Lw = params.Lr, params.Lw
    x, w = 3, 4
    r = [0] * (Lr - 1)
    u = [0] * (Lw - 1)
    xs, ws, qrs, qws, costs = [], [], [], [], []
    for d in demands:
        sr, su = sum(r), sum(u)
        qr, qw = orders(params, kind, S1, S2, x, w, sr, su)
        qr, qw = int(qr), int(qw)
        costs.append(params.h_w * (w + sr) + params.h_r * max(x - d, 0) + params.p * max(d - x, 0))
        xs.append(x)
        ws.append(w)
        qrs.append(qr)
        qws.append(qw)
        inflow_r = r.pop(0) if Lr > 1 else qr
        inflow_w = u.pop(0) if Lw > 1 else qw
        if Lr > 1:
            r.append(qr)
        if Lw > 1:
            u.append(qw)
        x = x - d + inflow_r
        w = w - qr + inflow_w
    tail = costs[burn:]
    return Trace(xs, ws, qrs, qws, costs, float(np.mean(tail)) if tail else 0.0)


def lag_share(orders_w: list, demands: list, lag: int, burn: int = 0) -> float:
    """Anteil der Perioden t >= burn, in denen die Bestellung des Lagers genau der Nachfrage von Periode t - lag entspricht."""
    ts = range(max(burn, lag), len(orders_w))
    if not ts:
        return 0.0
    return sum(1 for t in ts if orders_w[t] == demands[t - lag]) / len(ts)
