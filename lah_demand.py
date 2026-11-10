"""Nachfrage: Poisson-Verteilung mit gedeckeltem Ende und Faltung über mehrere Perioden (nur numpy und Standardbibliothek)."""

from math import exp, factorial

import numpy as np


def poisson_pmf(mean: float, dmax: int) -> np.ndarray:
    """Poisson(mean) auf 0..dmax; die Restwahrscheinlichkeit jenseits von dmax liegt auf dmax (die Summe ist genau 1)."""
    pmf = np.array([exp(-mean) * mean**k / factorial(k) for k in range(dmax + 1)])
    pmf[-1] += 1.0 - pmf.sum()
    return pmf


def convolve_n(pmf, n: int) -> np.ndarray:
    """Verteilung der Summe von n unabhängigen Perioden-Nachfragen (n = 0: die Einsverteilung auf 0)."""
    out = np.array([1.0])
    for _ in range(n):
        out = np.convolve(out, pmf)
    return out


def mean_of(pmf) -> float:
    pmf = np.asarray(pmf, dtype=float)
    return float((np.arange(len(pmf)) * pmf).sum())
