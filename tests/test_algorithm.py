"""Kern gegen Handrechnung und Gegenproben, die nichts aus der Rechenroutine übernehmen (DEMO-PLAYBOOK §1b und §12): Rasteranpassung, Zerlegung, DP, Politiken.

Die Einheiten am Handbeispiel stehen in tests/test_units.py, die Orakel in tests/test_oracle_lager_und_haendler.py."""

import numpy as np
import pytest

import lah_algorithm as A
import lah_chain as CH
import lah_policies as P


def test_fit_raster_reproduces_the_main_cell():
    joint = A.fit_raster(1, 1, 0.4, 9.0)
    assert (joint.decomposition.S1, joint.decomposition.S2) == (10, 13)
    assert (joint.params.xmin, joint.params.xmax, joint.params.W) == (-20, 12, 19) and joint.grown == 0
    assert joint.dp.g == pytest.approx(6.6012936, abs=1e-6) and joint.dp_chain.g == pytest.approx(joint.dp.g, abs=1e-6)
    assert joint.space.n_states == 33 * 20 == 660


def test_fit_raster_grows_a_too_narrow_raster_on_both_sides_and_in_the_warehouse():
    """Absichtlich zu enge Rasterstarts: x oben (xmax 8), x unten (xmin -3), Lager (W 8). Danach ist die Randmasse unter 1e-6 und der DP-Wert gleich dem der Hauptzelle."""
    for start in ((-20, 8, 19), (-3, 12, 19), (-20, 12, 8)):
        joint = A.fit_raster(1, 1, 0.4, 9.0, start=start)
        assert joint.grown >= 1, start
        c = joint.dp_chain
        assert max(c.edge_x_low, c.edge_x_high, c.edge_w) <= A.EDGE_TOLERANCE
        assert joint.dp.g == pytest.approx(6.6012936, abs=1e-5), start


def test_fit_raster_gives_up_with_an_error_when_the_raster_cannot_grow_enough():
    with pytest.raises(RuntimeError):
        A.fit_raster(1, 1, 0.4, 9.0, Q=1)           # Bestellgrenze 1: der Rückstand wächst ohne Ende, das Raster reicht nie


def test_dp_policy_on_reached_states_matches_the_echelon_rule_by_hand_in_one_state():
    joint = A.fit_raster(1, 1, 0.4, 9.0)
    space = joint.space
    s = int(np.flatnonzero((space.x == 7) & (space.w == 3))[0])               # Zustand der Hauptzelle (x = 7, w = 3): Echelon bestellt (3, 3)
    assert divmod(int(joint.dp.policy[s]), space.nq) == (3, 3)
    assert P.policy_actions(space, "echelon", 10, 13)[s] == joint.dp.policy[s]


def test_echelon_levels_beat_their_neighbours_on_the_exact_chain():
    joint = A.fit_raster(1, 1, 0.4, 9.0)
    base = CH.evaluate(joint.space, P.policy_actions(joint.space, "echelon", 10, 13)).g
    for s1, s2 in ((9, 13), (11, 13), (10, 12), (10, 14)):
        assert CH.evaluate(joint.space, P.policy_actions(joint.space, "echelon", s1, s2)).g > base
