"""Jede Zahl der README als Test (Tabelle „Zahl → Test“ in der README). Zählungen laufen über alle 48 gemessenen Zellen (data/lah_grid.json, erzeugt von tools/build_grid.py),
nicht über Stichproben; die Zellen selbst sind in tests/test_grid.py gegen eine Neuberechnung geprüft. Wo ein Satz eine Ausnahme hat, steht sie hier mit Namen."""

import numpy as np
import pytest

import lah_chain as CH
import lah_constants as C
import lah_decompose as D
import lah_dp
import lah_evaluation as ev
import lah_example as EX
import lah_policies as P
import lah_results as R
import lah_scaling as SC
from lah_model import Params, StateSpace

CELLS = R.load_grid()
MAIN = R.get_cell(1, 1, 0.4, 9)


def test_claim_decomposition_equals_joint_dp_in_48_of_48_cells():
    t = R.echelon_equals_dp()
    assert t["n"] == 48 and t["cost_equal"] == 48 and t["max_cost_diff"] < 1e-8              # bis auf die gedruckten fünf Stellen (größter Unterschied unter 1e-8)
    assert t["best_enum_same_cost"] == 48 and t["best_enum_same_levels"] == 48               # beste Echelon-Politik der Aufzählung (Kette, Fenster um die Zerlegung)
    assert t["formula_same_levels"] == 48                                                    # ... und der Aufzählung mit der Formel im weiten Fenster
    assert t["max_action_diff"] < 4e-11                                                      # Aktion der DP weicht höchstens in 3,4e-11 der Zeit ab


def test_claim_main_cell_costs_to_seven_digits():
    assert round(MAIN["g_dp"], 7) == 6.6012936 and round(MAIN["g_echelon"], 7) == 6.6012936
    assert round(MAIN["g_formula"], 5) == 6.60129                                           # Formel ohne Bestellgrenze: 6,6012914 (Unterschied 2,3e-6 durch die Grenze)


def test_claim_edge_masses_are_below_1e4():
    t = R.echelon_equals_dp()
    assert t["max_cap"] < 3.3e-5 and t["max_cap"] > 3.2e-5                                   # Bestellgrenze höchstens 3,2e-5 (3,23e-5)
    assert t["max_edge"] < 2e-9                                                              # Händlerbestand und Lager: 0 bis 2e-9
    assert max(c["rules"][k]["cap"] for c in CELLS for k in ("A", "B")) < 6e-7               # lokale Regeln: Bestellgrenze höchstens 6e-7
    assert MAIN["cap_dp"] < 3e-5
    assert max(c["rules"][k]["edge_w"] for c in CELLS for k in ("echelon", "A", "B", "C")) == 0.0       # Lagerbestand nie an der Rastergrenze
    assert max(c["rules"][k]["edge_x_low"] for c in CELLS for k in ("echelon", "C")) < 2e-9              # Händlerbestand unten: Echelon und Sicht C höchstens 1,7e-9
    assert 1.1e-5 < max(c["rules"]["A"]["edge_x_low"] for c in CELLS) < 1.3e-5                           # Sicht A: höchstens 1,2e-5 (größte Randmasse der lokalen Regeln, unter 1e-4)
    assert max(c["rules"]["B"]["edge_x_low"] for c in CELLS) < 8e-7


def test_claim_three_views_in_the_main_cell():
    assert round(R.rule_gap(MAIN, "A"), 2) == 11.51 and round(R.rule_gap(MAIN, "B"), 2) == 4.12 and abs(R.rule_gap(MAIN, "C")) < 1e-6
    assert (MAIN["rules"]["A"]["S1"], MAIN["rules"]["A"]["S2"]) == (10, 9) and (MAIN["rules"]["B"]["S1"], MAIN["rules"]["B"]["S2"]) == (10, 5)
    assert (MAIN["S1"], MAIN["S2"]) == (10, 13) and (MAIN["rules"]["C"]["S1"], MAIN["rules"]["C"]["S2"]) == (10, 3)


@pytest.mark.parametrize("lr,lw,a,b", [(1, 1, 11.51, 4.12), (1, 2, 9.88, 3.85), (2, 1, 8.93, 3.26), (2, 2, 7.78, 3.33)])
def test_claim_gaps_by_lead_times_at_h_w_0_4_p_9(lr, lw, a, b):
    cell = R.get_cell(lr, lw, 0.4, 9)
    assert round(R.rule_gap(cell, "A"), 2) == a and round(R.rule_gap(cell, "B"), 2) == b and abs(R.rule_gap(cell, "C")) < 1e-6


def test_claim_gap_ranges_over_all_48_cells():
    lo_a, hi_a = R.gap_range("A")
    lo_b, hi_b = R.gap_range("B")
    assert (round(lo_a, 2), round(hi_a, 2)) == (3.73, 20.94) and (round(lo_b, 2), round(hi_b, 2)) == (0.79, 11.27)
    assert sum(abs(R.rule_gap(c, "C")) < 1e-6 for c in CELLS) == 48
    assert R.echelon_equals_dp()["a_gt_b_gt_0"] == 48                                       # A > B > 0 in 48 von 48


def test_claim_gap_by_h_w_extremes():
    at = lambda hw, k: [R.rule_gap(c, k) for c in CELLS if c["h_w"] == hw]  # noqa: E731
    assert (round(min(at(0.2, "A")), 2), round(max(at(0.2, "A")), 2)) == (3.73, 6.95)
    assert (round(min(at(0.8, "A")), 2), round(max(at(0.8, "A")), 2)) == (11.50, 20.94)


def test_claim_trends_with_h_w_p_and_lead_times_with_named_exceptions():
    assert R.monotone_in_hw("A") == (12, 12) and R.monotone_in_hw("B") == (12, 12)
    ok, n, bad = R.monotone_in_p("A")
    assert (ok, n) == (15, 16) and bad == [(2, 1, 0.8)]
    g = lambda lr, lw, hw, p, k: R.rule_gap(R.get_cell(lr, lw, hw, p), k)  # noqa: E731
    assert (round(g(2, 1, 0.8, 9, "A"), 2), round(g(2, 1, 0.8, 19, "A"), 2)) == (13.35, 13.83)     # die Ausnahme: Lücke steigt von p 9 auf 19
    assert R.monotone_in_p("B")[:2] == (12, 16)                                              # B fällt nur in 12 von 16 Reihen (nicht gedeutet)
    ok_r, n_r, _ = R.lead_effect("A", "Lr")
    ok_w, n_w, bad_w = R.lead_effect("A", "Lw")
    assert (ok_r, n_r) == (24, 24) and (ok_w, n_w) == (23, 24)
    assert len(bad_w) == 1 and (bad_w[0]["Lr"], bad_w[0]["Lw"], bad_w[0]["h_w"], bad_w[0]["p"]) == (1, 1, 0.8, 4.0)
    assert (round(bad_w[0]["gap1"], 2), round(bad_w[0]["gap2"], 2)) == (20.69, 20.94)


def test_claim_view_A_at_warehouse_lead_time_1_equals_view_B_at_2_in_24_of_24_pairs():
    pairs = R.delay_pairs()
    assert len(pairs) == 24 and all(q["diff"] < 1e-9 for q in pairs) and max(q["diff"] for q in pairs) < 1e-12


def test_claim_levels_by_lead_times():
    assert R.s1_independent_of_lw() == (24, 24) and R.s1_grows_with_lr() == (24, 24) and R.spread_grows_with_lw() == (24, 24)
    assert (min(c["S1"] for c in CELLS), max(c["S1"] for c in CELLS)) == (8, 17)
    spread = [c["S2"] - c["S1"] for c in CELLS]
    assert (min(spread), max(spread)) == (0, 9)
    zero = [(c["Lr"], c["Lw"], c["h_w"], c["p"]) for c in CELLS if c["S2"] == c["S1"]]
    assert zero == [(1, 1, 0.8, 4.0), (2, 1, 0.8, 4.0)]                                      # das Lager reicht nur durch


def test_claim_cost_breakdown_and_stocks_in_the_main_cell():
    e, a, b = MAIN["rules"]["echelon"], MAIN["rules"]["A"], MAIN["rules"]["B"]
    assert [round(e[k], 4) for k in ("hold_w", "hold_r", "short")] == [1.4688, 3.5083, 1.6241]
    assert [round(a[k], 4) for k in ("hold_w", "hold_r", "short")] == [2.4000, 3.8942, 1.0669]
    assert [round(b[k], 4) for k in ("hold_w", "hold_r", "short")] == [2.0000, 3.9089, 0.9641]
    assert (round(e["mean_w"], 2), round(a["mean_w"], 2), round(b["mean_w"], 2)) == (3.67, 6.00, 5.00)
    assert (round(100 * e["unfilled"], 1), round(100 * a["unfilled"], 1), round(100 * b["unfilled"], 1)) == (35.3, 10.6, 11.1)
    assert sum(e[k] for k in ("hold_w", "hold_r", "short")) == pytest.approx(e["g"], abs=1e-9)


def test_claim_view_A_holds_more_warehouse_stock_than_echelon_in_48_of_48_cells():
    assert R.warehouse_stock_larger() == (48, 48)


def test_claim_order_quotas_main_cell_seed_11():
    run = ev.run(ev.Settings(seed=11), length=C.RUN_LENGTH, burn=C.BURN_IN)
    assert run.lag_echelon[1] > 0.999                                                        # Echelon bestellt die Nachfrage der letzten Periode (Quote 1,000)
    assert 0.82 < run.lag_local[2] < 0.88 and run.lag_local[1] < 0.25                        # Sicht A: die der vorletzten (0,85 bei 20.000 Perioden, 0,848 bei 200.000)
    assert abs(run.mean_echelon - MAIN["rules"]["echelon"]["g"]) < 0.15 and abs(run.mean_local - MAIN["rules"]["A"]["g"]) < 0.15   # Simulation streut um den exakten Wert


def test_claim_state_counts_and_growth_factors():
    rows = {(r.Lr, r.Lw): r for r in SC.table()}
    assert [rows[k].dp_states for k in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2))] == [760, 12768, 208544, 11760, 197568]
    assert rows[(4, 4)].dp_states == 11_437_365_184
    ratios = [rows[(a, b + 1)].dp_states / rows[(a, b)].dp_states for a in (1, 2, 3, 4) for b in (1, 2, 3)]
    ratios += [rows[(a + 1, b)].dp_states / rows[(a, b)].dp_states for a in (1, 2, 3) for b in (1, 2, 3, 4)]
    assert len(ratios) == 24 and f"{min(ratios):.1f}" == "14.6" and f"{max(ratios):.1f}" == "16.9"
    lw_steps = [rows[(a, b + 1)].dp_states / rows[(a, b)].dp_states for a in (1, 2, 3, 4) for b in (1, 2, 3)]
    assert 15.5 <= min(lw_steps) and max(lw_steps) < 17.0                                    # je Periode Lieferzeit des Lagers: 15,5- bis 17,0-fach
    assert rows[(1, 1)].decomposition_steps == 133_940 and rows[(4, 4)].decomposition_steps == 1_218_854
    assert [(r.Lr, r.Lw) for r in rows.values() if r.dp_solved] == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)]


def test_claim_hand_example():
    ex = EX.report()
    assert (ex["decomposition"].S1, ex["decomposition"].S2) == (3, 4) and ex["decomposition"].g == pytest.approx(31 / 16, abs=1e-12)
    assert ex["L1"][:5] == [7.0, 3.75, 1.5, 0.75, 1.0] and ex["G2"][1:4] == [2.375, 1.9375, 2.25]
    assert [(round(ex["local"][k]["gap"], 2), ex["local"][k]["S1"], ex["local"][k]["Sw"]) for k in "ABC"] == [(18.05, 3, 3), (16.13, 3, 2), (0.0, 3, 1)]
    assert {k: v[1] for k, v in ex["orders"].items()} == {"A": 1, "B": 3, "C": 4}
    assert len(ex["chain"]) == 6 and ex["g_chain"] == pytest.approx(31 / 16, abs=1e-12)


@pytest.mark.parametrize("Q,expect_cap,zer_gap", [(5, 0.26, 1.2845), (8, 0.020, 0.0)])
def test_claim_order_limit_sweep_main_cell(Q, expect_cap, zer_gap):
    """Bestellgrenze 5 und 8 (die übrigen Grenzen: scripts/qsweep.py, Messung der README): beste Echelon-Politik = DP; die Niveaus der Zerlegung verlieren bei Grenze 5 +1,28 %, bei 8 nichts."""
    zer = D.solve(Params())
    space = StateSpace(Params(Q=Q, xmin=-26, xmax=12, W=26))
    dp = lah_dp.value_iteration(space)
    res = CH.evaluate(space, dp.policy)
    assert res.cap == pytest.approx(expect_cap, abs=0.005)
    best = min(CH.evaluate(space, P.policy_actions(space, "echelon", s1, s2)).g for s1 in range(zer.S1 - 1, zer.S1 + 3) for s2 in range(zer.S2 - 1, zer.S2 + 3))
    assert best == pytest.approx(dp.g, abs=1e-6)
    g_zer = CH.evaluate(space, P.policy_actions(space, "echelon", zer.S1, zer.S2)).g
    assert 100 * (g_zer / dp.g - 1) == pytest.approx(zer_gap, abs=0.001)


def test_claim_levels_of_the_preset_cells_are_those_named_in_the_text():
    assert [(c["S1"], c["S2"]) for c in R.preset_cells()] == [(10, 13), (10, 17), (14, 17), (12, 13), (10, 14), (11, 15)]
