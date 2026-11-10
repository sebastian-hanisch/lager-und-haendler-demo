"""Exakte Auswertung einer festen Politik auf der Markov-Kette des Zustandsrasters (numpy und scipy.sparse).

Übergangsmatrix je Politik, stationäre Verteilung auf der vom Anfangszustand erreichbaren Menge (direkt für kleine Mengen, sonst Potenziteration der trägen Kette),
mittlere Kosten je Periode und Randmassen (Anteil der Zeit am Rand des Rasters bzw. an der Bestellgrenze)."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import breadth_first_order

from lah_model import StateSpace


@dataclass(frozen=True)
class ChainResult:
    g: float
    pi: np.ndarray               # stationäre Verteilung über alle Rasterzustände
    n_reachable: int
    edge_x_low: float            # Anteil der Zeit mit Händlerbestand am unteren Rand des Rasters
    edge_x_high: float           # ... am oberen Rand
    edge_w: float                # ... Lagerbestand an der Rastergrenze W
    cap: float                   # ... mit einer Bestellung an der Bestellgrenze Q
    parts: dict                  # Kostenanteile je Periode: hold_w, hold_r, short; dazu mean_w (mittlerer Lagerbestand)


def transition_matrix(space: StateSpace, policy: np.ndarray) -> sp.csr_matrix:
    """Dünne Übergangsmatrix der Kette unter der Politik (Aktionsindex je Zustand); Bestände jenseits des Rasters werden auf den Rand gekappt."""
    p = space.params
    qr, qw = policy // space.nq, policy % space.nq
    x0, w_next, r_next, u_next = space.successor_parts(qr, qw)
    x_next = np.clip(x0[:, None] + p.xmin - space.d[None, :], p.xmin, p.xmax) - p.xmin
    idx = x_next * space.nw + np.clip(w_next, 0, p.W)[:, None]
    for v in r_next + u_next:
        idx = idx * space.nq + v[:, None]
    n, n_d = space.n_states, len(space.d)
    return sp.csr_matrix((np.tile(space.pmf, n), (np.repeat(np.arange(n), n_d), idx.ravel())), shape=(n, n))


def stationary(space: StateSpace, policy: np.ndarray, method: str = "auto", tol: float = 1e-14, maxit: int = 40000):
    """(pi, erreichbare Menge). method 'direct': dünnes lineares System; 'power': Potenziteration der trägen Kette; 'auto' wählt nach der Größe."""
    P = transition_matrix(space, policy)
    reach = np.sort(breadth_first_order(P, space.start_state(), directed=True, return_predecessors=False))
    sub = P[reach][:, reach]
    n = len(reach)
    if method == "auto":
        method = "direct" if n < 600 else "power"
    if method == "direct":
        a = (sub.T - sp.identity(n, format="csr")).tocsr()
        system = sp.vstack([a[:-1], sp.csr_matrix(np.ones((1, n)))]).tocsc()
        rhs = np.zeros(n)
        rhs[-1] = 1.0
        pi_reach = spla.spsolve(system, rhs)
    else:
        lazy = ((sub + sp.identity(n, format="csr")) * 0.5).T.tocsr()
        pi_reach = np.full(n, 1.0 / n)
        for _ in range(maxit):
            nxt = lazy @ pi_reach
            if np.abs(nxt - pi_reach).sum() < tol:
                pi_reach = nxt
                break
            pi_reach = nxt
        pi_reach = pi_reach / pi_reach.sum()
    pi = np.zeros(space.n_states)
    pi[reach] = pi_reach
    return pi, reach


def evaluate(space: StateSpace, policy: np.ndarray, method: str = "auto") -> ChainResult:
    """Mittlere Kosten je Periode der Politik (exakt, Potenz- bzw. Direktlösung) mit Randmassen und Kostenanteilen."""
    p = space.params
    pi, reach = stationary(space, policy, method=method)
    qr, qw = policy // space.nq, policy % space.nq
    parts = dict(
        hold_w=float(pi @ space.cost_hold_w), hold_r=float(pi @ space.cost_hold_r), short=float(pi @ space.cost_short),
        mean_w=float(pi @ space.w),
    )
    return ChainResult(
        g=float(pi @ space.cost), pi=pi, n_reachable=len(reach),
        edge_x_low=float(pi[space.x == p.xmin].sum()), edge_x_high=float(pi[space.x == p.xmax].sum()),
        edge_w=float(pi[space.w == p.W].sum()), cap=float(pi[(qr == p.Q) | (qw == p.Q)].sum()), parts=parts,
    )


def unfilled_share(space: StateSpace, pi: np.ndarray, S1: int) -> float:
    """Anteil der Zeit, in dem das Lager die Anforderung des Händlers (Auffüllung auf S1) nicht voll bedienen kann."""
    req = np.maximum(S1 - (space.x + space.sr), 0)
    return float(pi[req > space.w].sum())
