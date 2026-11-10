"""Zerlegung nach Clark und Scarf: zwei eindimensionale Rechnungen, nacheinander gekoppelt (nur numpy und Standardbibliothek).

Echelon-Bestand einer Stufe = Bestand der Stufe plus alles, was hinter ihr liegt (auf dem Weg und beim Händler). Kostensätze auf dem Echelon-Bestand:
Händlerstufe h_r - h_w je Stück Händlerbestand und p + h_w je fehlendes Stück; Lagerstufe h_w je Stück Echelon-Bestand. Die Funktionen sind die eigene Herleitung
für das Zeitschema dieser Demo (Kosten auf dem Bestand nach der Nachfrage); die Struktur (erst die Händlerstufe, dann das Lager mit der von der Händlerstufe
induzierten Kostenfunktion) entspricht Chao und Zhou (2006), Abschnitt 2.1."""

from dataclasses import dataclass

import numpy as np

from lah_demand import convolve_n, mean_of
from lah_model import Params


@dataclass(frozen=True)
class Decomposition:
    S1: int          # Niveau der Händlerstufe (Echelon-Bestand nach der Bestellung)
    S2: int          # Niveau der Lagerstufe
    g: float         # mittlere Kosten je Periode der Politik (S1, S2), geschlossene Formel ohne Bestellgrenze
    L1_min: float
    ties1: int       # Zahl der y mit (numerisch) gleich kleinen Kosten; 1 = eindeutig
    ties2: int


def retailer_cost(params: Params, y) -> np.ndarray:
    """L1(y): erwartete Kosten der Händlerstufe bei Echelon-Bestand y nach der Bestellung, Nachfrage über Lr + 1 Perioden."""
    d_retail = convolve_n(params.pmf(), params.Lr + 1)
    y = np.asarray(y)
    z = y[..., None] - np.arange(len(d_retail))
    hold = params.h_r - params.h_w
    short = params.p + params.h_w
    return (d_retail * (hold * np.maximum(z, 0) + short * np.maximum(-z, 0))).sum(-1)


def warehouse_cost(params: Params, y2, S1: int) -> np.ndarray:
    """G2(y2; S1): Kosten der Lagerstufe bei Echelon-Bestand y2; innen die Händlerstufe mit min(S1, y2 - Nachfrage über Lw Perioden)."""
    pmf = params.pmf()
    d_wh = convolve_n(pmf, params.Lw)
    y2 = np.asarray(y2)
    inner = np.minimum(S1, y2[..., None] - np.arange(len(d_wh)))
    return params.h_w * (y2 - (params.Lw + 1) * mean_of(pmf)) + (d_wh * retailer_cost(params, inner)).sum(-1)


def solve(params: Params, lo: int = -40, hi: int = 140) -> Decomposition:
    """S1 = Minimum von L1, S2 = Minimum von G2 mit diesem S1; bei Gleichstand der kleinste Wert (np.argmin)."""
    ys = np.arange(lo, hi + 1)
    v1 = retailer_cost(params, ys)
    S1 = int(ys[np.argmin(v1)])
    v2 = warehouse_cost(params, ys, S1)
    S2 = int(ys[np.argmin(v2)])
    return Decomposition(
        S1=S1, S2=S2, g=float(v2.min()), L1_min=float(v1.min()),
        ties1=int((np.abs(v1 - v1.min()) < 1e-12).sum()), ties2=int((np.abs(v2 - v2.min()) < 1e-12).sum()),
    )


def echelon_cost(params: Params, S1: int, S2: int) -> float:
    """Mittlere Kosten je Periode der Echelon-Politik (S1, S2) nach der geschlossenen Formel (ohne Bestellgrenze)."""
    return float(warehouse_cost(params, np.array([S2]), S1)[0])
