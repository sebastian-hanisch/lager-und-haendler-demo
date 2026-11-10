"""Bestellregeln: Echelon-Base-Stock und die lokale Regel in drei Sichten des Lagers (nur numpy und Standardbibliothek).

Der Händler bestellt in allen Regeln bis zum Niveau S1 auf seiner Bestandsposition (x + unterwegs), höchstens so viel, wie das Lager hat (und höchstens Q).
Das Lager bestellt beim Lieferanten:
  echelon  bis S2 auf dem Echelon-Bestand w + unterwegs zum Händler + x + unterwegs zum Lager (der Echelon-Bestand sieht alles hinter dem Lager)
  A        bis Sw auf w + unterwegs zum Lager; das Lager entscheidet gleichzeitig mit dem Händler und kennt nur seinen eigenen Bestand
  B        bis Sw auf w - qr + unterwegs zum Lager; das Lager sieht, was es in dieser Periode an den Händler ausliefert
  C        bis Sw auf w - Anforderung + unterwegs zum Lager; das Lager sieht die volle Bestellung des Händlers, auch den Teil, den es nicht liefern kann
Bei den lokalen Regeln ist der zweite Parameter das Niveau Sw des Lagers, beim Echelon-Base-Stock das Echelon-Niveau S2."""

import numpy as np

from lah_model import Params, StateSpace

KINDS = ("echelon", "A", "B", "C")
KIND_LABEL = {
    "echelon": "Echelon-Base-Stock (Zerlegung)",
    "A": "lokal, Sicht A: Lager sieht nur seinen Bestand",
    "B": "lokal, Sicht B: Lager sieht, was es liefert",
    "C": "lokal, Sicht C: Lager sieht die volle Bestellung",
}


def _lo_hi(x):
    """Skalare mit eingebautem min/max (schnell in der Simulation), Felder mit numpy."""
    return (min, max) if isinstance(x, (int, np.integer)) else (np.minimum, np.maximum)


def retailer_request(S1, x, sr):
    """Anforderung des Händlers beim Lager: Auffüllung auf S1 (nie negativ)."""
    return _lo_hi(x)[1](S1 - (x + sr), 0)


def orders(params: Params, kind: str, S1, S2, x, w, sr, su):
    """(qr, qw) für Zustand(e) (x, w, sr, su); Skalare oder gleich lange Felder. S2 ist das Echelon-Niveau (echelon) bzw. das Lagerniveau Sw (A, B, C)."""
    mn, mx = _lo_hi(x)
    req = retailer_request(S1, x, sr)
    qr = mn(mn(req, w), params.Q)
    if kind == "echelon":
        qw = S2 - (w + sr + x + su)
    elif kind == "A":
        qw = S2 - (w + su)
    elif kind == "B":
        qw = S2 - (w - qr + su)
    elif kind == "C":
        qw = S2 - (w - req + su)
    else:
        raise ValueError(f"unbekannte Regel {kind!r}")
    qw = mn(mx(qw, 0), params.Q)
    return qr, qw


def echelon_orders(params: Params, S1, S2, x, w, sr, su):
    return orders(params, "echelon", S1, S2, x, w, sr, su)


def local_orders(params: Params, kind: str, S1, Sw, x, w, sr, su):
    if kind not in ("A", "B", "C"):
        raise ValueError(f"keine lokale Regel: {kind!r}")
    return orders(params, kind, S1, Sw, x, w, sr, su)


def policy_actions(space: StateSpace, kind: str, S1: int, S2: int) -> np.ndarray:
    """Aktionsindex qr * (Q + 1) + qw der Regel für jeden Zustand des Rasters."""
    qr, qw = orders(space.params, kind, S1, S2, space.x, space.w, space.sr, space.su)
    return (qr * space.nq + qw).astype(np.int64)
