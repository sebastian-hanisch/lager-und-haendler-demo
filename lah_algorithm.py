"""Kern der Demo: gemeinsame Rechnung auf einem Raster, das wächst, bis die DP-Politik den Rand nicht mehr nutzt (nur numpy und scipy).

Die Regeln selbst stehen in lah_decompose (Zerlegung), lah_dp (gemeinsame Rückwärtsrechnung), lah_policies (Bestellregeln) und lah_chain (exakte Auswertung);
dieses Modul setzt sie für eine Zelle zusammen und prüft beidseitig die Ränder (eine DP, die ein bekanntes Optimum unterbietet, ist ein Randalarm: bei zu engem Raster
geht Bestand am Rand kostenlos verloren)."""

from dataclasses import dataclass

import lah_chain as CH
import lah_constants as C
import lah_decompose as D
import lah_dp
from lah_model import Params, StateSpace

EDGE_TOLERANCE = 1e-6    # Randmasse (Anteil der Zeit am Rand des Rasters), ab der das Raster wächst
MAX_GROWTHS = 6


@dataclass(frozen=True)
class JointResult:
    params: Params
    space: StateSpace
    decomposition: D.Decomposition
    dp: lah_dp.DPResult
    dp_chain: CH.ChainResult     # die DP-Politik exakt auf der Kette ausgewertet (Randmassen, Zeitanteile)
    grown: int                   # Zahl der Rastervergrößerungen


def fit_raster(Lr: int, Lw: int, h_w: float, p: float, Q: int = C.Q_MAX, start: tuple | None = None) -> JointResult:
    """Zerlegung, Raster und gemeinsame DP einer Zelle. Start: x von -20 bis S1 + 2, Lagerbestand 0 bis S2 - S1 + 16; das Raster wächst unten um 8, oben um 4, im Lager um 6,
    solange die Randmasse der DP-Politik über 1e-6 liegt. `start` = (xmin, xmax, W) überschreibt den Anfang (Test mit absichtlich zu engem Raster)."""
    dec = D.solve(Params(Lr=Lr, Lw=Lw, h_w=h_w, p=p, Q=Q))
    xmin, xmax, W = start if start is not None else (-20, dec.S1 + 2, dec.S2 - dec.S1 + 16)
    grown = 0
    for _ in range(MAX_GROWTHS + 1):
        params = Params(Lr=Lr, Lw=Lw, h_w=h_w, p=p, Q=Q, xmin=xmin, xmax=xmax, W=W)
        space = StateSpace(params)
        dp = lah_dp.value_iteration(space)
        chain = CH.evaluate(space, dp.policy)
        if chain.edge_x_low > EDGE_TOLERANCE:
            xmin -= 8
        elif chain.edge_x_high > EDGE_TOLERANCE:
            xmax += 4
        elif chain.edge_w > EDGE_TOLERANCE:
            W += 6
        else:
            return JointResult(params, space, dec, dp, chain, grown)
        grown += 1
    raise RuntimeError(f"das Raster reicht auch nach {MAX_GROWTHS} Vergrößerungen nicht (Lieferzeit {Lr}/{Lw}, h_w {h_w}, p {p})")
