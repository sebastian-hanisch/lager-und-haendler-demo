"""Orakel-Tests (ORACLE-PLAYBOOK §5): jede Kernroutine gegen einen anderen Rechenweg.

O1 gemeinsame DP gegen Echelon-Politik (Kosten und Aktionen auf der DP-Verteilung)
O2 Zerlegung (geschlossene Formel) gegen Aufzählung der Paare (S1, S2) auf der exakten Kette
O3 Simulation (eigene Schleife und eigene Regelfunktionen im Test, eingefrorener Nachfragestrom) gegen die exakte Auswertung
O4 Handbeispiel exakt mit `fractions`, ohne den Code der Demo (L1, G2, Kette, 31/16)
O5 die neu geschriebene DP gegen die Werte der älteren Rechnung aus der Vorarbeit (eingefroren, andere Bauart: dünne Matrix je Aktion)
O6 direkte Lösung der stationären Verteilung gegen Potenziteration
O7 Sicht C gegen Echelon-Politik; Sicht A bei Lagerlieferzeit 1 gegen Sicht B bei Lagerlieferzeit 2
Der große Lauf (48 Zellen) steht in tools/build_grid.py und in tests/test_claims.py (über die vorgerechnete Datei)."""

from fractions import Fraction as F

import numpy as np
import pytest

import lah_chain as CH
import lah_constants as C
import lah_decompose as D
import lah_dp
import lah_policies as P
import lah_simulate as S
from lah_model import Params, StateSpace


def _cell_space(lr, lw, h_w, p):
    base = Params(Lr=lr, Lw=lw, h_w=h_w, p=float(p), Q=C.Q_MAX)
    dec = D.solve(base)
    params = Params(Lr=lr, Lw=lw, h_w=h_w, p=float(p), Q=C.Q_MAX, xmin=-20, xmax=dec.S1 + 2, W=dec.S2 - dec.S1 + 16)
    return base, dec, StateSpace(params)


# --- O1 ------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lr,lw,h_w,p", [(1, 1, 0.4, 9), (1, 1, 0.2, 4), (1, 1, 0.8, 19), (1, 1, 0.6, 9), (1, 2, 0.4, 9)])
def test_O1_joint_dp_equals_echelon_policy_of_the_decomposition(lr, lw, h_w, p):
    base, dec, space = _cell_space(lr, lw, h_w, p)
    dp = lah_dp.value_iteration(space)
    assert dp.converged
    pe = P.policy_actions(space, "echelon", dec.S1, dec.S2)
    ech = CH.evaluate(space, pe)
    dp_chain = CH.evaluate(space, dp.policy)
    assert dp_chain.g == pytest.approx(dp.g, abs=1e-6)                      # die Politik der DP kostet, was die DP meldet
    assert ech.g == pytest.approx(dp.g, abs=1e-6)                           # Echelon-Politik = Optimum
    differing = dp_chain.pi[(dp.policy != pe) & (dp_chain.pi > 1e-12)].sum()
    assert differing < 1e-8                                                  # gleiche Aktion auf (praktisch) allen erreichten Zuständen
    assert dp_chain.edge_x_low < 1e-6 and dp_chain.edge_x_high < 1e-6 and dp_chain.edge_w < 1e-6   # Raster beidseitig ausreichend


# --- O2 ------------------------------------------------------------------------------------------------------------------------------------

def test_O2_formula_equals_exact_chain_for_many_pairs_and_the_decomposition_is_the_enumeration_optimum():
    base, dec, space = _cell_space(1, 1, 0.4, 9)
    worst = 0.0
    for s1 in (dec.S1 - 2, dec.S1, dec.S1 + 2):
        for s2 in (dec.S2 - 2, dec.S2, dec.S2 + 3):
            gf = D.echelon_cost(base, s1, s2)
            gc = CH.evaluate(space, P.policy_actions(space, "echelon", s1, s2)).g
            worst = max(worst, abs(gf - gc))
    assert worst < 2e-5                         # Unterschied = Einfluss der Bestellgrenze (Zeitanteil an der Grenze 2,6e-5)
    best_formula, best_chain = (1e9, None), (1e9, None)
    for s1 in range(1, dec.S1 + 9):
        for s2 in range(s1, s1 + 27):
            gf = D.echelon_cost(base, s1, s2)
            if gf < best_formula[0] - 1e-12:
                best_formula = (gf, (s1, s2))
    for s1 in range(dec.S1 - 2, dec.S1 + 3):
        for s2 in range(max(s1, dec.S2 - 3), dec.S2 + 4):
            gc = CH.evaluate(space, P.policy_actions(space, "echelon", s1, s2)).g
            if gc < best_chain[0] - 1e-12:
                best_chain = (gc, (s1, s2))
    assert best_formula[1] == best_chain[1] == (dec.S1, dec.S2) == (10, 13)


def test_O2_formula_matches_chain_with_longer_lead_times():
    base, dec, space = _cell_space(2, 1, 0.4, 9)
    for s1, s2 in ((dec.S1, dec.S2), (dec.S1 - 1, dec.S2 + 1), (dec.S1 + 1, dec.S2 - 1)):
        gc = CH.evaluate(space, P.policy_actions(space, "echelon", s1, s2)).g
        assert D.echelon_cost(base, s1, s2) == pytest.approx(gc, abs=1e-5)


# --- O3 ------------------------------------------------------------------------------------------------------------------------------------

def _policy_python(kind, S1, S2, Q):
    """Eigene Regelfunktion des Tests (reines Python, Lieferzeit 1/1): (x, w) -> (qr, qw)."""
    def f(x, w):
        req = max(S1 - x, 0)
        qr = min(req, w, Q)
        if kind == "echelon":
            qw = S2 - (w + x)
        elif kind == "A":
            qw = S2 - w
        else:
            qw = S2 - (w - qr)
        return qr, min(max(qw, 0), Q)
    return f


def _simulate_python(f, demands, params, burn=1000):
    x, w, costs = 3, 4, []
    for d in demands:
        qr, qw = f(x, w)
        costs.append(params.h_w * w + params.h_r * max(x - d, 0) + params.p * max(d - x, 0))
        x, w = x - d + qr, w - qr + qw
    return np.array(costs[burn:])


@pytest.mark.parametrize("kind,levels", [("echelon", (10, 13)), ("A", (10, 9)), ("B", (10, 5))])
def test_O3_simulation_matches_exact_value_within_four_standard_errors(main_cell_params, kind, levels):
    space = StateSpace(main_cell_params)
    exact = CH.evaluate(space, P.policy_actions(space, kind, *levels)).g
    f = _policy_python(kind, levels[0], levels[1], C.Q_MAX)
    means, variances = [], []
    for seed in (101, 202, 303, 404):
        demands = S.demand_stream(main_cell_params.pmf(), 61000, seed)
        batches = _simulate_python(f, demands, main_cell_params).reshape(20, -1).mean(axis=1)
        means.append(batches.mean())
        variances.append(batches.var(ddof=1) / 20)
    se = np.sqrt(np.sum(variances)) / 4
    assert abs(np.mean(means) - exact) < 4 * se, (np.mean(means), exact, se)


def test_O3_module_simulation_equals_the_test_simulation_on_the_same_stream(main_cell_params):
    demands = S.demand_stream(main_cell_params.pmf(), 3000, 7)
    for kind, levels in (("echelon", (10, 13)), ("A", (10, 9)), ("B", (10, 5))):
        trace = S.simulate(main_cell_params, kind, *levels, demands, burn=0)
        mine = _simulate_python(_policy_python(kind, levels[0], levels[1], C.Q_MAX), demands, main_cell_params, burn=0)
        assert np.allclose(trace.cost, mine)


def test_O3_splitmix_reference_vector():
    """SplitMix64 mit Seed 0: erste Ausgabe 0xE220A8397B1DCDAF (veröffentlichter Referenzwert); die Demo skaliert (z >> 11) / 2^53."""
    assert S.SplitMix(0).next_uniform() == (0xE220A8397B1DCDAF >> 11) / float(1 << 53)


# --- O4 ------------------------------------------------------------------------------------------------------------------------------------

DEMAND = {0: F(1, 4), 1: F(1, 2), 2: F(1, 4)}
H_HOLD, H_SHORT = F(1, 2), F(7, 2)        # Echelon-Sätze der Händlerstufe: 1 − 1/2 und 3 + 1/2


def _convolve(n):
    dist = {0: F(1)}
    for _ in range(n):
        new = {}
        for k, pk in dist.items():
            for d, pd in DEMAND.items():
                new[k + d] = new.get(k + d, F(0)) + pk * pd
        dist = new
    return dist


def _L1(y):
    return sum(pr * (H_HOLD * max(y - k, 0) + H_SHORT * max(k - y, 0)) for k, pr in _convolve(2).items())


def _G2(y, S1):
    return F(1, 2) * (y - 2) + sum(pr * _L1(min(S1, y - d)) for d, pr in DEMAND.items())


def _stationary_exact(S1, S2, Q=4):
    """Kette der Politik (S1, S2) mit exakten Brüchen: erreichbare Zustände ab (3, 4), Gauß-Elimination, mittlere Kosten."""
    def move(x, w, d):
        qr = min(max(S1 - x, 0), w, Q)
        qw = min(max(S2 - (w + x), 0), Q)
        return (x - d + qr, w - qr + qw)
    states, todo = [], [(3, 4)]
    while todo:
        s = todo.pop()
        if s in states:
            continue
        states.append(s)
        todo.extend(move(*s, d) for d in DEMAND)
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    # pi (P − I) = 0 und Summe pi = 1; letzte Gleichung ersetzt durch die Normierung
    a = [[F(0)] * n for _ in range(n)]
    for s in states:
        for d, pr in DEMAND.items():
            a[index[move(*s, d)]][index[s]] += pr
    for i in range(n):
        a[i][i] -= 1
    a[n - 1] = [F(1)] * n
    b = [F(0)] * (n - 1) + [F(1)]
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r][col] != 0)
        a[col], a[piv], b[col], b[piv] = a[piv], a[col], b[piv], b[col]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col] / a[col][col]
                a[r] = [u - f * v for u, v in zip(a[r], a[col])]
                b[r] -= f * b[col]
    pi = {s: b[index[s]] / a[index[s]][index[s]] for s in states}
    cost = lambda x, w: F(1, 2) * w + sum(pr * (max(x - d, 0) + 3 * max(d - x, 0)) for d, pr in DEMAND.items())  # noqa: E731
    return pi, sum(pr * cost(*s) for s, pr in pi.items())


def test_O4_hand_example_exactly_with_fractions():
    assert [_L1(y) for y in range(5)] == [F(7), F(15, 4), F(3, 2), F(3, 4), F(1)]
    S1 = min(range(0, 8), key=_L1)
    S2 = min(range(0, 10), key=lambda y: _G2(y, S1))
    assert (S1, S2) == (3, 4) and _G2(S2, S1) == F(31, 16)
    assert [_G2(y, 3) for y in (3, 4, 5)] == [F(19, 8), F(31, 16), F(9, 4)]
    pi, g = _stationary_exact(3, 4)
    assert g == F(31, 16)
    recurrent = {s: pr for s, pr in pi.items() if pr != 0}
    assert recurrent == {(0, 2): F(1, 16), (1, 1): F(3, 16), (1, 2): F(1, 8), (2, 1): F(3, 8), (2, 2): F(1, 16), (3, 1): F(3, 16)}


def test_O4_demo_code_agrees_with_the_exact_fractions(hand_params, hand_space):
    dec = D.solve(hand_params, lo=-10, hi=20)
    assert dec.g == pytest.approx(float(F(31, 16)), abs=1e-12)
    res = CH.evaluate(hand_space, P.policy_actions(hand_space, "echelon", 3, 4), method="direct")
    pi, g = _stationary_exact(3, 4)
    assert res.g == pytest.approx(float(g), abs=1e-12)
    for (x, w), pr in pi.items():
        s = int(np.flatnonzero((hand_space.x == x) & (hand_space.w == w))[0])
        assert res.pi[s] == pytest.approx(float(pr), abs=1e-12)
    for s1, s2 in ((3, 3), (3, 5), (3, 6), (2, 4), (4, 4)):          # auch Politiken, die nicht optimal sind
        assert CH.evaluate(hand_space, P.policy_actions(hand_space, "echelon", s1, s2), method="direct").g == pytest.approx(float(_stationary_exact(s1, s2)[1]), abs=1e-12)


def test_O4_enumeration_of_all_echelon_pairs_finds_3_4_and_the_dp_agrees(hand_space):
    best = min(((s1, s2) for s1 in range(1, 6) for s2 in range(s1, 9)), key=lambda c: _stationary_exact(*c)[1])
    assert best == (3, 4)
    dp = lah_dp.value_iteration(hand_space, tol=1e-12)
    assert dp.g == pytest.approx(31 / 16, abs=1e-9)
    pe = P.policy_actions(hand_space, "echelon", 3, 4)
    pi = CH.evaluate(hand_space, pe, method="direct").pi
    assert pi[(dp.policy != pe) & (pi > 1e-12)].sum() < 1e-9


# --- O5 ------------------------------------------------------------------------------------------------------------------------------------

def test_O5_new_dp_equals_the_older_dense_matrix_values_of_the_preliminary_work():
    """Eingefrorene Werte der älteren Rechnung (je Aktion eine dünne Matrix, andere Bauart): Mittel 3, Bestellgrenze 8, Lagerbestand bis 18, x von −30 bis 25, Nachfrage bis 14."""
    params = Params(Q=8, W=18, xmin=-30, xmax=25, dmax=14)
    space = StateSpace(params)
    dp = lah_dp.value_iteration(space)
    assert dp.g == pytest.approx(6.6048, abs=5e-5)
    assert CH.evaluate(space, P.policy_actions(space, "echelon", 10, 13)).g == pytest.approx(6.6048, abs=5e-5)
    assert CH.evaluate(space, P.policy_actions(space, "A", 10, 9)).g == pytest.approx(7.3612, abs=5e-5)


# --- O6 ------------------------------------------------------------------------------------------------------------------------------------

def test_O6_direct_solution_equals_power_iteration_for_lead_time_1_2():
    base, dec, space = _cell_space(1, 2, 0.4, 9)
    pol = P.policy_actions(space, "echelon", dec.S1, dec.S2)
    assert CH.evaluate(space, pol, method="direct").g == pytest.approx(CH.evaluate(space, pol, method="power").g, abs=1e-9)


# --- O7 ------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lr,lw", [(1, 1), (1, 2)])
def test_O7_view_C_with_Sw_equal_S2_minus_S1_is_the_echelon_policy(lr, lw):
    base, dec, space = _cell_space(lr, lw, 0.4, 9)
    worst_cost, worst_share = 0.0, 0.0
    for s1 in (dec.S1 - 1, dec.S1, dec.S1 + 1):
        for sw in (dec.S2 - dec.S1 - 1, dec.S2 - dec.S1, dec.S2 - dec.S1 + 2):
            pc, pe = P.policy_actions(space, "C", s1, sw), P.policy_actions(space, "echelon", s1, s1 + sw)
            rc, re = CH.evaluate(space, pc), CH.evaluate(space, pe)
            worst_cost = max(worst_cost, abs(rc.g - re.g))
            worst_share = max(worst_share, float(re.pi[pc != pe].sum()))
    assert worst_cost < 1e-9 and worst_share < 1e-9


def test_O7_view_A_with_lead_time_1_costs_the_same_as_view_B_with_lead_time_2():
    pairs = [(s1, sw) for s1 in (9, 10) for sw in (5, 7, 9)]
    params1 = Params(Lr=1, Lw=1, xmin=-20, xmax=14, W=26)
    params2 = Params(Lr=1, Lw=2, xmin=-20, xmax=14, W=26)
    sp1, sp2 = StateSpace(params1), StateSpace(params2)
    for s1, sw in pairs:
        ga = CH.evaluate(sp1, P.policy_actions(sp1, "A", s1, sw)).g
        gb = CH.evaluate(sp2, P.policy_actions(sp2, "B", s1, sw)).g
        assert ga == pytest.approx(gb, abs=1e-9), (s1, sw)
