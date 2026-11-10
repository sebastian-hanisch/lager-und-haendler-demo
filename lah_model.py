"""Modell: Lieferant -> Lager -> Händler -> Kunde, Rückstand beim Händler (nur numpy und Standardbibliothek).

Zeitablauf einer Periode (Zustand zu ihrem Beginn): Händler-Nettobestand x (negativ = Rückstand), Lagerbestand w, Unterwegs zum Händler
(Lr - 1 Posten r), Unterwegs zum Lager (Lw - 1 Posten u). Der Händler bestellt qr <= w beim Lager, das Lager bestellt qw beim
Lieferanten (unbegrenzter Vorrat); die Bestellung des Händlers kommt nach Lr, die des Lagers nach Lw Perioden an (bei Lieferzeit 1 am Ende der
laufenden Periode, nach der Nachfrage). Kosten der Periode: h_w * (w + Summe r) + h_r * (x - D)^+ + p * (D - x)^+.

`h_r` ist der Lagerkostensatz für den Bestand des Händlers insgesamt, `h_w` der Satz für das Lager und die Ware auf dem Weg zum Händler
(der Echelon-Satz der Händlerstufe ist h_r - h_w). Bestellkosten sind 0.
"""

from dataclasses import dataclass

import numpy as np

from lah_demand import poisson_pmf


@dataclass(frozen=True, kw_only=True)
class Params:
    mean: float = 3.0
    h_w: float = 0.4
    h_r: float = 1.0
    p: float = 9.0
    Lr: int = 1               # Lieferzeit der Händlerbestellung beim Lager
    Lw: int = 1               # Lieferzeit der Lagerbestellung beim Lieferanten
    Q: int = 13               # Bestellgrenze je Stufe
    W: int = 24               # Rastergrenze des Lagerbestands
    xmin: int = -22           # Rastergrenzen des Händler-Nettobestands
    xmax: int = 14
    dmax: int = 18
    dist: tuple | None = None  # eigene Nachfrageverteilung (Handbeispiel); sonst Poisson(mean) bis dmax

    def pmf(self) -> np.ndarray:
        if self.dist is not None:
            return np.array(self.dist, dtype=float)
        return poisson_pmf(self.mean, self.dmax)


def period_cost(params: Params, x: int, w: int, sr: int = 0) -> float:
    """Erwartete Kosten einer Periode im Zustand (x, w) bei Summe r = sr unterwegs zum Händler."""
    pmf = params.pmf()
    shortage_and_holding = sum(
        pr * (params.h_r * max(x - d, 0) + params.p * max(d - x, 0)) for d, pr in enumerate(pmf)
    )
    return params.h_w * (w + sr) + float(shortage_and_holding)


def step(params: Params, state: tuple, qr: int, qw: int, d: int) -> tuple:
    """Folgezustand (x', w', r', u') bei Bestellung (qr, qw) und Nachfrage d; ohne Rastergrenzen.

    state = (x, w, r, u) mit r, u als Tupeln (r[0] kommt als Nächstes beim Händler an, u[0] beim Lager)."""
    x, w, r, u = state
    inflow_r = r[0] if params.Lr > 1 else qr
    inflow_w = u[0] if params.Lw > 1 else qw
    r_next = (tuple(r[1:]) + (qr,)) if params.Lr > 1 else ()
    u_next = (tuple(u[1:]) + (qw,)) if params.Lw > 1 else ()
    return x - d + inflow_r, w - qr + inflow_w, r_next, u_next


class StateSpace:
    """Zustandsraster (x, w, r..., u...) mit Indexarithmetik und der Faltungsmatrix über die Nachfrage."""

    def __init__(self, params: Params):
        self.params = params
        nr, nu = params.Lr - 1, params.Lw - 1
        self.nr, self.nu = nr, nu
        self.nx, self.nw, self.nq = params.xmax - params.xmin + 1, params.W + 1, params.Q + 1
        dims = [self.nx, self.nw] + [self.nq] * (nr + nu)
        self.dims = dims
        self.n_states = int(np.prod(dims, dtype=np.int64))
        grid = np.indices(dims, dtype=np.int32).reshape(len(dims), -1)
        self.x = grid[0].astype(np.int64) + params.xmin
        self.w = grid[1].astype(np.int64)
        self.r = grid[2 : 2 + nr].astype(np.int64)
        self.u = grid[2 + nr :].astype(np.int64)
        zeros = np.zeros(self.n_states, dtype=np.int64)
        self.sr = self.r.sum(axis=0) if nr else zeros
        self.su = self.u.sum(axis=0) if nu else zeros
        self.pmf = params.pmf()
        self.d = np.arange(len(self.pmf))
        self.n_actions = self.nq * self.nq
        # erwartete Kosten je Zustand, nach Anteilen getrennt (Lager und Weg zum Händler halten, Händler halten, Rückstand)
        hold_r = np.zeros(self.n_states)
        short = np.zeros(self.n_states)
        for dd, pr in zip(self.d, self.pmf):
            hold_r += pr * params.h_r * np.maximum(self.x - dd, 0)
            short += pr * params.p * np.maximum(dd - self.x, 0)
        self.cost_hold_w = params.h_w * (self.w + self.sr)
        self.cost_hold_r = hold_r
        self.cost_short = short
        self.cost = self.cost_hold_w + hold_r + short
        # Faltungsmatrix entlang x: conv[x0] = sum_d pmf(d) * h[clip(x0 - d)]; x0 reicht bis xmax + Q
        n_x0 = self.nx + params.Q
        conv = np.zeros((n_x0, self.nx))
        for i in range(n_x0):
            for dd, pr in zip(self.d, self.pmf):
                j = min(max(i + params.xmin - dd, params.xmin), params.xmax) - params.xmin
                conv[i, j] += pr
        self.conv = conv

    def encode(self, x_index, w_next, r_next, u_next):
        """Index des Zustands aus Händler-Index (nach der Faltung x0), Lagerbestand (auf die Rastergrenze gekappt) und den Pipelines."""
        i = x_index * self.nw + np.clip(w_next, 0, self.params.W)
        for v in list(r_next) + list(u_next):
            i = i * self.nq + v
        return i

    def successor_parts(self, qr, qw):
        """(x0 - xmin als Index vor der Nachfrage, Lagerbestand', Pipeline r', Pipeline u') für Bestellungen qr, qw (Skalare oder Felder)."""
        p = self.params
        x0 = self.x + (self.r[0] if self.nr else qr)
        w_next = self.w - qr + (self.u[0] if self.nu else qw)
        ones = np.ones(self.n_states, dtype=np.int64)
        r_next = (list(self.r[1:]) + [qr * ones]) if self.nr else []
        u_next = (list(self.u[1:]) + [qw * ones]) if self.nu else []
        return x0 - p.xmin, w_next, r_next, u_next

    def next_index(self, qr: int, qw: int):
        """Index des Folgezustands in der um die Faltung erweiterten Wertetafel, für feste Bestellung (qr, qw)."""
        x0, w_next, r_next, u_next = self.successor_parts(qr, qw)
        return self.encode(x0, w_next, r_next, u_next)

    def start_state(self) -> int:
        """Anfangszustand der Verläufe: Händlerbestand 3, Lagerbestand 4 (auf das Raster gekappt), Pipelines leer."""
        p = self.params
        x, w = min(3, p.xmax), min(4, p.W)
        hit = np.flatnonzero((self.x == x) & (self.w == w) & (self.sr == 0) & (self.su == 0))
        return int(hit[0])
