"""Gemeinsame Fixtures: das Handbeispiel (Lieferzeit 1/1, Nachfrage 0, 1, 2 mit 1/4, 1/2, 1/4), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lah_constants as C  # noqa: E402
from lah_model import Params, StateSpace  # noqa: E402


@pytest.fixture(scope="session")
def hand_params():
    """h_Händler 1, h_Lager 0,5, p = 3, Bestellgrenze 4; Nachfrage 0, 1, 2 mit 1/4, 1/2, 1/4. Handrechnung: S1 = 3, S2 = 4, g = 31/16 = 1,9375."""
    return Params(**C.EXAMPLE)


@pytest.fixture(scope="session")
def hand_space(hand_params):
    return StateSpace(hand_params)


@pytest.fixture(scope="session")
def main_cell_params():
    """Hauptzelle: Lieferzeit 1/1, h_w 0,4, p 9, Poisson Mittel 3, Raster wie in der Messreihe."""
    return Params(Lr=1, Lw=1, h_w=0.4, p=9.0, Q=C.Q_MAX, xmin=-20, xmax=12, W=19)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: dauert länger als zehn Sekunden (lastbehaftet), ausdrücklich markiert")
