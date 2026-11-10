"""Gemeinsame Rückwärtsrechnung (relative Wertiteration) über den Zustand beider Bestände und aller Pipelines (nur numpy).

Mittlere Kosten je Periode im Unendlichen: h(s) + g = min_a [ c(s) + E h(s') ], mit h(Bezugszustand) = 0. Der Erwartungswert über die Nachfrage läuft als
Faltung entlang des Händlerbestands x (eine Matrixmultiplikation je Iteration), die Aktionen (qr, qw) als Schleife über alle (Q + 1)^2 Paare mit qr <= w."""

from dataclasses import dataclass

import numpy as np

from lah_model import StateSpace

CACHE_LIMIT = 250_000   # bis zu dieser Zustandszahl werden die Folgeindizes aller Aktionen zwischengespeichert (rund 200 MB); darüber neu berechnet


@dataclass(frozen=True)
class DPResult:
    g: float                 # mittlere Kosten je Periode
    policy: np.ndarray       # Aktionsindex qr * (Q + 1) + qw je Zustand; bei Gleichstand gewinnt die kleinere Aktion
    h: np.ndarray            # relative Wertefunktion
    iterations: int
    converged: bool


def value_iteration(space: StateSpace, tol: float = 1e-9, maxit: int = 4000) -> DPResult:
    S, nq = space.n_states, space.nq
    cache = {}
    use_cache = S <= CACHE_LIMIT

    def successor(a):
        if a in cache:
            return cache[a]
        qr, qw = divmod(a, nq)
        idx = space.next_index(qr, qw).astype(np.int32)
        if use_cache:
            cache[a] = idx
        return idx

    feasible = [space.w >= (a // nq) for a in range(space.n_actions)] if S <= 5000 else None
    ref = space.start_state()
    h = np.zeros(S)
    converged = False
    diff = np.zeros(S)
    it = 0
    for it in range(1, maxit + 1):
        smoothed = (space.conv @ h.reshape(space.nx, -1)).ravel()
        best = np.full(S, np.inf)
        for a in range(space.n_actions):
            q = space.cost + smoothed[successor(a)]
            ok = feasible[a] if feasible is not None else (space.w >= a // nq)
            np.minimum(best, np.where(ok, q, np.inf), out=best)
        diff = best - h
        h = best - best[ref]
        if diff.max() - diff.min() < tol:
            converged = True
            break
    g = 0.5 * (diff.max() + diff.min())
    smoothed = (space.conv @ h.reshape(space.nx, -1)).ravel()
    best = np.full(S, np.inf)
    arg = np.zeros(S, dtype=np.int64)
    for a in range(space.n_actions):
        q = np.where(space.w >= a // nq, space.cost + smoothed[successor(a)], np.inf)
        better = q < best - 1e-12
        best = np.where(better, q, best)
        arg = np.where(better, a, arg)
    return DPResult(g=float(g), policy=arg, h=h, iterations=it, converged=converged)
