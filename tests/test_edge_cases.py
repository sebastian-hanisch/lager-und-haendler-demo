"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): konstante Nachfrage, sehr kleine Fehlmengenkosten, Bestellgrenze 1, Gleichheit der Lagerkostensätze,
ein zu enges Raster (beidseitige Randprüfung), das kleinste Raster."""

import numpy as np
import pytest

import lah_chain as CH
import lah_decompose as D
import lah_dp
import lah_policies as P
from lah_model import Params, StateSpace


def test_constant_demand_has_zero_variance_cost_2_times_h_w():
    """Nachfrage immer 2, Lieferzeit 1/1: S1 = 4 (Bestand genau für zwei Perioden), S2 = 6; stationär x = 2, w = 2, Kosten je Periode h_w · 2 = 0,8 (von Hand)."""
    params = Params(dist=(0.0, 0.0, 1.0), Q=6, xmin=-4, xmax=8, W=10)
    dec = D.solve(params, lo=0, hi=30)
    assert (dec.S1, dec.S2) == (4, 6) and dec.g == pytest.approx(2 * 0.4)
    space = StateSpace(params)
    assert lah_dp.value_iteration(space, tol=1e-12).g == pytest.approx(0.8, abs=1e-9)
    assert CH.evaluate(space, P.policy_actions(space, "echelon", 4, 6)).g == pytest.approx(0.8, abs=1e-9)


def test_very_small_shortage_cost_gives_low_levels_and_still_solves():
    params = Params(p=0.1, Lr=1, Lw=1)
    dec = D.solve(params)
    assert dec.S1 < D.solve(Params()).S1                          # kritisches Verhältnis (p + h_w) / (h_r + p) ist kleiner als bei p = 9
    assert np.isfinite(dec.g) and dec.g > 0


def test_order_limit_1_is_hit_almost_always_and_the_edge_check_reports_it():
    """Bestellgrenze 1 bei Nachfrage 3: die Politik kann nicht nachkommen, der Anteil an der Bestellgrenze nahe 1 und der Rückstand läuft ans Raster."""
    params = Params(Q=1, W=6, xmin=-30, xmax=8)
    space = StateSpace(params)
    pol = P.policy_actions(space, "echelon", 4, 6)
    res = CH.evaluate(space, pol)
    assert res.cap > 0.9
    assert res.edge_x_low > 1e-3                                  # Rückstand wächst bis zum unteren Rand: die Randprüfung schlägt an


def test_equal_holding_rates_make_the_retailer_stage_non_increasing():
    """h_w = h_r (nicht gemessen, Regler endet bei 0,8): der Echelon-Satz der Händlerstufe ist 0, L1 fällt nur noch, S1 ist nicht eindeutig oder liegt am Rand des Suchfensters."""
    params = Params(h_w=1.0, h_r=1.0)
    values = D.retailer_cost(params, np.arange(0, 60))
    assert np.all(np.diff(values) <= 1e-12)


def test_too_narrow_raster_is_reported_by_the_edge_masses():
    """Absichtlich zu enges Raster (x höchstens 4): die DP nutzt den Rand, die Randmasse oben ist deutlich über 1e-3 (beide Ränder prüfen)."""
    params = Params(Lr=1, Lw=1, xmin=-20, xmax=4, W=19)
    space = StateSpace(params)
    dp = lah_dp.value_iteration(space)
    res = CH.evaluate(space, dp.policy)
    assert max(res.edge_x_high, res.edge_x_low, res.edge_w) > 1e-3


def test_two_demand_values_and_lead_time_1_1_smallest_raster_runs():
    params = Params(dist=(0.5, 0.5), Q=2, xmin=-2, xmax=3, W=3)
    space = StateSpace(params)
    dp = lah_dp.value_iteration(space, tol=1e-12)
    assert dp.converged and np.isfinite(dp.g)
    assert CH.evaluate(space, dp.policy, method="direct").g == pytest.approx(dp.g, abs=1e-8)


def test_probabilities_of_the_chain_sum_to_one(main_cell_params):
    space = StateSpace(main_cell_params)
    res = CH.evaluate(space, P.policy_actions(space, "A", 10, 9))
    assert res.pi.sum() == pytest.approx(1.0, abs=1e-12) and res.pi.min() >= 0.0
