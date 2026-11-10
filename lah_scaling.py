"""Rechenlast: Zahl der Zustände der gemeinsamen Rechnung gegen den Aufwand der Zerlegung (nur numpy und Standardbibliothek)."""

from dataclasses import dataclass

import lah_decompose as D
from lah_model import Params

XMIN_RASTER = -20      # untere Rastergrenze des Händlerbestands in der Rechenlast-Tafel
X_ABOVE_S1 = 7         # obere Rastergrenze: S1 + 7 (bei S1 + 2 nutzte die DP bei 1/3 den Rand aus: g 7,47804 statt 7,48026)
W_ABOVE = 16           # Lagerbestandsgrenze: S2 - S1 + 16
DP_SOLVED_MAX_SUM = 4  # die gemeinsame DP wurde nur gerechnet, wenn Lr + Lw <= 4
# Größenordnung der Rechenzeit der gemeinsamen DP (Messung der Vorarbeit, lastbehaftet, nie als feste Zahl lesen)
DP_TIME_CLASS = {
    (1, 1): "unter 0,1 s", (1, 2): "bis 0,3 s", (2, 1): "bis 0,3 s",
    (1, 3): "rund 5 bis 10 s", (2, 2): "rund 5 bis 10 s", (3, 1): "rund 5 bis 10 s",
}


@dataclass(frozen=True)
class ScalingRow:
    Lr: int
    Lw: int
    S1: int
    S2: int
    g: float                 # Kosten der Zerlegung (geschlossene Formel)
    dp_states: int           # Zustände des Rasters der gemeinsamen DP
    decomposition_steps: int  # Rechenschritte der Zerlegung (Summanden beider eindimensionaler Rechnungen auf 181 Werten von y)
    dp_solved: bool          # in der Vorarbeit gerechnet (sonst nur gezählt)
    dp_time: str


def state_count(Lr: int, Lw: int, S1: int, S2: int, Q: int = 13) -> int:
    """Zustände des Rasters: Händlerbestand von -20 bis S1 + 7, Lagerbestand 0 bis S2 - S1 + 16, je Pipelineplatz 0 bis Q."""
    nx = (S1 + X_ABOVE_S1) - XMIN_RASTER + 1
    nw = S2 - S1 + W_ABOVE + 1
    return nx * nw * (Q + 1) ** (Lr + Lw - 2)


def decomposition_steps(Lr: int, Lw: int, values: int = 181, mean: float = 3.0, dmax: int = 18) -> int:
    """Zahl der Multiplikationen der Zerlegung: L1 über `values` Werte von y mit der Nachfrage über Lr + 1 Perioden, G2 mit Lw-Perioden-Nachfrage und L1 darin."""
    len_d1 = (Lr + 1) * dmax + 1
    len_dw = Lw * dmax + 1
    return values * len_d1 + values * len_dw * len_d1


def table(max_lead: int = 4, **overrides) -> list[ScalingRow]:
    rows = []
    for Lr in range(1, max_lead + 1):
        for Lw in range(1, max_lead + 1):
            params = Params(Lr=Lr, Lw=Lw, **overrides)
            dec = D.solve(params)
            rows.append(ScalingRow(
                Lr=Lr, Lw=Lw, S1=dec.S1, S2=dec.S2, g=dec.g,
                dp_states=state_count(Lr, Lw, dec.S1, dec.S2, params.Q),
                decomposition_steps=decomposition_steps(Lr, Lw, mean=params.mean),
                dp_solved=Lr + Lw <= DP_SOLVED_MAX_SUM,
                dp_time=DP_TIME_CLASS.get((Lr, Lw), "nicht gerechnet"),
            ))
    return rows
