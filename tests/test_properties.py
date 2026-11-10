"""Eigenschaften und Zweig-Tests: Invarianten der Rechnung (Warenerhaltung, Konvexität, Zeitanteile), und jede Variante weicht auf mindestens einer Instanz ab
(Nullspalte = Variante läuft nie, DEMO-PLAYBOOK §0)."""

import numpy as np
import pytest

import lah_chain as CH
import lah_constants as C
import lah_decompose as D
import lah_evaluation as ev
import lah_policies as P
import lah_results as R
import lah_simulate as S
from lah_model import Params, StateSpace


# --- Warenerhaltung ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lr,lw", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("kind", P.KINDS)
def test_stock_is_conserved_in_every_step(lr, lw, kind):
    """Echelon-Bestand E_t = Lager + Händler + Weg zum Händler + Weg zum Lager; E_(t+1) = E_t - D_t + qw_t (nichts geht verloren, nichts entsteht)."""
    params = Params(Lr=lr, Lw=lw, h_w=0.4, p=9.0, Q=C.Q_MAX)
    demands = S.demand_stream(params.pmf(), 400, 5)
    trace = S.simulate(params, kind, 10, 13 if kind == "echelon" else 9, demands)

    def echelon(t):
        sr = sum(trace.qr[t - k] for k in range(1, lr) if t - k >= 0)
        su = sum(trace.qw[t - k] for k in range(1, lw) if t - k >= 0)
        return trace.x[t] + trace.w[t] + sr + su

    for t in range(len(demands) - 1):
        assert echelon(t + 1) == echelon(t) - demands[t] + trace.qw[t]
        assert 0 <= trace.qr[t] <= min(trace.w[t], C.Q_MAX) and 0 <= trace.qw[t] <= C.Q_MAX


def test_orders_to_the_retailer_never_exceed_the_warehouse_stock():
    params = Params()
    xs = np.arange(-15, 20)
    for kind in P.KINDS:
        qr, qw = P.orders(params, kind, 10, 9, xs, np.full(xs.size, 3), np.zeros(xs.size, dtype=int), np.zeros(xs.size, dtype=int))
        assert np.all(qr <= 3) and np.all(qr >= 0) and np.all(qw >= 0)


# --- Konvexität und Unabhängigkeiten ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lr,lw", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_decomposition_costs_are_convex_in_S2(lr, lw):
    params = Params(Lr=lr, Lw=lw)
    dec = D.solve(params)
    ys = np.arange(0, 60)
    second = np.diff(D.warehouse_cost(params, ys, dec.S1), 2)
    assert np.all(second > -1e-10)


@pytest.mark.parametrize("lr", C.LEADS)
def test_S1_does_not_depend_on_the_warehouse_lead_time(lr):
    for h_w in C.H_W_GRID:
        for p in C.P_GRID:
            a = D.solve(Params(Lr=lr, Lw=1, h_w=h_w, p=float(p)))
            b = D.solve(Params(Lr=lr, Lw=2, h_w=h_w, p=float(p)))
            assert a.S1 == b.S1


def test_S1_grows_with_the_retailer_lead_time_and_S2_minus_S1_with_the_warehouse_lead_time():
    for h_w in C.H_W_GRID:
        for p in C.P_GRID:
            assert D.solve(Params(Lr=2, h_w=h_w, p=float(p))).S1 > D.solve(Params(Lr=1, h_w=h_w, p=float(p))).S1
            d1, d2 = D.solve(Params(Lw=1, h_w=h_w, p=float(p))), D.solve(Params(Lw=2, h_w=h_w, p=float(p)))
            assert d2.S2 - d2.S1 > d1.S2 - d1.S1


def test_live_decomposition_equals_the_stored_levels_in_all_48_cells():
    for c in R.load_grid():
        dec = D.solve(Params(Lr=c["Lr"], Lw=c["Lw"], h_w=c["h_w"], p=c["p"]))
        assert (dec.S1, dec.S2) == (c["S1"], c["S2"]) and dec.g == pytest.approx(c["g_formula"], abs=1e-12)


# --- Zweig-Tests: jede Variante ändert das Ergebnis --------------------------------------------------------------------------------------

def test_every_control_changes_the_result():
    base = ev.analyse(ev.Settings())
    changed = [
        ev.analyse(ev.Settings(lr=2)), ev.analyse(ev.Settings(lw=2)), ev.analyse(ev.Settings(hw=0.8)), ev.analyse(ev.Settings(p=19)),
        ev.analyse(ev.Settings(d1=2)), ev.analyse(ev.Settings(d2=-2)),
    ]
    for other in changed:
        assert (other.S1, other.S2, round(other.g_dp, 9), round(other.g_levels, 9)) != (base.S1, base.S2, round(base.g_dp, 9), round(base.g_levels, 9))
    assert len({round(base.gaps[k], 6) for k in C.VIEWS}) == 3               # jede Sicht hat in der Hauptzelle eine andere Lücke
    assert base.gaps["C"] < base.gaps["B"] < base.gaps["A"]


def test_each_view_orders_something_different_in_some_state():
    params = Params()
    seen = {k: int(P.orders(params, k, 10, 9, 5, 4, 0, 0)[1]) for k in C.VIEWS}
    assert len(set(seen.values())) == 3, seen                               # Anforderung 5, Lager 4: A, B, C bestellen verschieden


def test_offsets_change_the_simulation_run():
    a = ev.run(ev.Settings(), length=3000, burn=500)
    b = ev.run(ev.Settings(d1=3), length=3000, burn=500)
    c = ev.run(ev.Settings(view="B"), length=3000, burn=500)
    assert a.mean_echelon != b.mean_echelon and a.mean_local != c.mean_local
    assert a.demands == b.demands                                           # gleicher Nachfragestrom für beide Regeln und beide Einstellungen


def test_different_seeds_give_different_demand_streams_and_the_same_seed_the_same():
    pmf = Params().pmf()
    assert S.demand_stream(pmf, 200, 1) != S.demand_stream(pmf, 200, 2)
    assert S.demand_stream(pmf, 200, 1) == S.demand_stream(pmf, 200, 1)


# --- Zeitanteile ---------------------------------------------------------------------------------------------------------------------------

def test_time_shares_sum_to_one_for_every_rule(main_cell_params):
    space = StateSpace(main_cell_params)
    for kind, levels in (("echelon", (10, 13)), ("A", (10, 9)), ("B", (10, 5)), ("C", (10, 3))):
        res = CH.evaluate(space, P.policy_actions(space, kind, *levels))
        assert res.pi.sum() == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= CH.unfilled_share(space, res.pi, 10) <= 1.0


def test_lag_share_hand_example():
    assert S.lag_share([0, 9, 0, 1], [9, 0, 1, 2], lag=1) == 1.0            # Bestellung t = Nachfrage t - 1
    assert S.lag_share([0, 9, 0, 1], [9, 0, 1, 2], lag=2, burn=2) == 0.0
    assert S.lag_share([], [], lag=1) == 0.0
