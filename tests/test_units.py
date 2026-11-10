"""Einheiten gegen Handrechnung am Handbeispiel (Plan, Abschnitt „Handbeispiel von Hand“): jede Zahl steht hier als Rechnung, nicht als Ausgabe des Codes.

Nachfrage D = 0, 1, 2 mit 1/4, 1/2, 1/4; Lieferzeit 1/1; h_Händler 1, h_Lager 0,5, p = 3; Echelon-Sätze: Händlerstufe 0,5 je Stück Bestand und 3,5 je fehlendes Stück,
Lagerstufe 0,5 je Stück Echelon-Bestand."""

import numpy as np
import pytest

import lah_chain as CH
import lah_decompose as D
import lah_demand as demand
import lah_dp
import lah_policies as P
import lah_scaling as SC
from lah_model import Params, StateSpace, period_cost, step


def test_convolve_n_two_periods_is_1_4_6_4_1_over_16(hand_params):
    got = demand.convolve_n(hand_params.pmf(), 2)
    assert got == pytest.approx(np.array([1, 4, 6, 4, 1]) / 16)


def test_convolve_n_zero_periods_is_the_point_mass_on_zero():
    assert demand.convolve_n(np.array([0.25, 0.5, 0.25]), 0).tolist() == [1.0]


def test_poisson_pmf_sums_to_one_and_has_the_right_mean():
    pmf = demand.poisson_pmf(3.0, 18)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-15)
    assert demand.mean_of(pmf) == pytest.approx(3.0, abs=1e-8)
    assert pmf[0] == pytest.approx(np.exp(-3.0))


def test_retailer_cost_L1_hand_values(hand_params):
    # L1(y) = sum_k P(k) [0,5 (y-k)^+ + 3,5 (k-y)^+] mit P = (1, 4, 6, 4, 1)/16
    expected = {0: 3.5 * 32 / 16, 1: 0.5 / 16 + 3.5 * 17 / 16, 2: 0.5 * 6 / 16 + 3.5 * 6 / 16, 3: 0.5 * 17 / 16 + 3.5 / 16, 4: 0.5 * 32 / 16}
    assert expected == {0: 7.0, 1: 3.75, 2: 1.5, 3: 0.75, 4: 1.0}
    for y, value in expected.items():
        assert float(D.retailer_cost(hand_params, np.array(y))) == pytest.approx(value, abs=1e-12)


def test_warehouse_cost_G2_hand_values(hand_params):
    # G2(y) = 0,5 (y - 2) + 1/4 L1(min(3, y)) + 1/2 L1(min(3, y-1)) + 1/4 L1(min(3, y-2))
    L1 = {0: 7.0, 1: 3.75, 2: 1.5, 3: 0.75, 4: 1.0}
    g = lambda y: 0.5 * (y - 2) + 0.25 * L1[min(3, y)] + 0.5 * L1[min(3, y - 1)] + 0.25 * L1[min(3, y - 2)]  # noqa: E731
    assert g(3) == pytest.approx(2.375) and g(4) == pytest.approx(1.9375) and g(5) == pytest.approx(2.25)
    for y in (3, 4, 5):
        assert float(D.warehouse_cost(hand_params, np.array([y]), 3)[0]) == pytest.approx(g(y), abs=1e-12)


def test_solve_returns_S1_3_S2_4_and_g_31_over_16(hand_params):
    dec = D.solve(hand_params, lo=-10, hi=20)
    assert (dec.S1, dec.S2) == (3, 4)
    assert dec.g == pytest.approx(31 / 16, abs=1e-12)
    assert (dec.ties1, dec.ties2) == (1, 1)         # beide Minima eindeutig


def test_S1_is_the_quantile_of_the_ratio_3_5_over_4(hand_params):
    cdf = np.cumsum(demand.convolve_n(hand_params.pmf(), 2))
    ratio = 3.5 / (3.5 + 0.5)
    assert ratio == 0.875 and int(np.flatnonzero(cdf >= ratio)[0]) == 3 == D.solve(hand_params, lo=-10, hi=20).S1


def test_period_cost_hand_values(hand_params):
    assert period_cost(hand_params, 0, 2) == pytest.approx(4.0)     # 0,5·2 + 3·E[D] = 1 + 3
    assert period_cost(hand_params, 3, 1) == pytest.approx(2.5)     # 0,5·1 + E(3 − D)^+ = 0,5 + 2


def test_step_hand_value():
    params = Params(Lr=1, Lw=1)
    assert step(params, (1, 2, (), ()), qr=2, qw=1, d=0) == (3, 1, (), ())


def test_step_with_longer_lead_times_moves_the_pipeline():
    params = Params(Lr=2, Lw=3)
    x, w, r, u = step(params, (5, 6, (4,), (2, 3)), qr=1, qw=7, d=2)
    assert (x, w, r, u) == (5 - 2 + 4, 6 - 1 + 2, (1,), (3, 7))


def test_echelon_orders_hand_values(hand_params):
    assert P.echelon_orders(hand_params, 3, 4, 0, 2, 0, 0) == (2, 2)
    assert P.echelon_orders(hand_params, 3, 4, 1, 2, 0, 0) == (2, 1)
    assert P.echelon_orders(hand_params, 3, 4, 3, 1, 0, 0) == (0, 0)


def test_local_orders_three_views_in_state_0_2_with_Sw_3(hand_params):
    # Anforderung 3, Lager hat 2 und liefert 2: A: 3 − 2 = 1; B: 3 − (2 − 2) = 3; C: 3 − (2 − 3) = 4
    assert [int(P.local_orders(hand_params, k, 3, 3, 0, 2, 0, 0)[1]) for k in "ABC"] == [1, 3, 4]


def test_orders_vectorized_equal_scalar(hand_params):
    xs, ws = np.meshgrid(np.arange(-3, 5), np.arange(0, 7))
    for kind in P.KINDS:
        qr_v, qw_v = P.orders(hand_params, kind, 3, 4, xs.ravel(), ws.ravel(), np.zeros(xs.size, dtype=int), np.zeros(xs.size, dtype=int))
        for i, (x, w) in enumerate(zip(xs.ravel(), ws.ravel())):
            assert (int(qr_v[i]), int(qw_v[i])) == tuple(int(v) for v in P.orders(hand_params, kind, 3, 4, int(x), int(w), 0, 0))


def test_unknown_rule_is_an_error(hand_params):
    with pytest.raises(ValueError):
        P.orders(hand_params, "D", 3, 4, 0, 2, 0, 0)
    with pytest.raises(ValueError):
        P.local_orders(hand_params, "echelon", 3, 4, 0, 2, 0, 0)


def test_stationary_distribution_of_the_echelon_chain_is_1_16_3_16_1_8_3_8_1_16_3_16(hand_space):
    pol = P.policy_actions(hand_space, "echelon", 3, 4)
    res = CH.evaluate(hand_space, pol, method="direct")
    expected = {(0, 2): 1 / 16, (1, 1): 3 / 16, (1, 2): 1 / 8, (2, 1): 3 / 8, (2, 2): 1 / 16, (3, 1): 3 / 16}
    for (x, w), prob in expected.items():
        s = int(np.flatnonzero((hand_space.x == x) & (hand_space.w == w))[0])
        assert res.pi[s] == pytest.approx(prob, abs=1e-12)
    assert res.pi.sum() == pytest.approx(1.0, abs=1e-12) and np.count_nonzero(res.pi > 1e-12) == 6
    assert res.g == pytest.approx(31 / 16, abs=1e-12)
    assert res.n_reachable == 13


def test_joint_dp_on_the_hand_example_gives_31_over_16(hand_space):
    dp = lah_dp.value_iteration(hand_space, tol=1e-12)
    assert dp.converged and dp.g == pytest.approx(31 / 16, abs=1e-9)


def test_state_count_hand_example_is_13_times_7(hand_space):
    assert hand_space.n_states == 13 * 7 == 91 and hand_space.n_actions == 25


def test_scaling_state_count_matches_the_raster_formula():
    # Händlerbestand −20 bis S1 + 7, Lagerbestand 0 bis S2 − S1 + 16, 14 Werte je Pipelineplatz: 1/1 mit S1 = 10, S2 = 13
    assert SC.state_count(1, 1, 10, 13) == (10 + 7 + 20 + 1) * (13 - 10 + 16 + 1) == 38 * 20 == 760
    assert SC.state_count(1, 2, 10, 17) == 38 * 24 * 14 == 12768


def test_scaling_table_has_16_rows_and_marks_unsolved_ones():
    rows = SC.table()
    assert len(rows) == 16
    by = {(r.Lr, r.Lw): r for r in rows}
    assert by[(1, 1)].dp_states == 760 and by[(4, 4)].dp_states == 11_437_365_184
    assert by[(2, 2)].dp_solved and not by[(1, 4)].dp_solved and not by[(3, 2)].dp_solved
