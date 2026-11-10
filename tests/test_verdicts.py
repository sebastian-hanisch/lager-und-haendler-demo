"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import lah_evaluation as ev
import lah_scaling as SC


def test_number_formatting_uses_decimal_comma_and_sign():
    assert ev.de(6.60129, 5) == "6,60129" and ev.de(0.5, 1, sign=True) == "+0,5" and ev.de(-2.345, 2, sign=True) == "-2,35"
    assert ev.pct(11.514, 1, sign=True) == "+11,5 %" and ev.pct(0.0, 2) == "0,00 %"


def test_decomposition_verdict_success_when_equal_within_1e5():
    kind, text = ev.verdict_decomposition(6.60129, 6.601295, 10, 13, 48, 48)
    assert kind == "success"
    assert "S1 = 10, S2 = 13" in text and "6,60130" in text and "6,60129" in text and "48 von 48" in text and "Kein Unterschied auf fünf Stellen" in text


def test_decomposition_verdict_threshold_is_1e5_exclusive():
    assert ev.verdict_decomposition(5.0, 5.000009, 3, 4, 47, 48)[0] == "success"
    kind, text = ev.verdict_decomposition(5.0, 5.00002, 3, 4, 47, 48)
    assert kind == "warning" and "+0,00002" in text and "47 von 48" in text


def test_view_verdict_A_names_gap_and_the_delay_count():
    kind, text = ev.verdict_view("A", 11.51, 24, 24)
    assert kind == "warning" and "+11,51 %" in text and "Sicht A" in text and "24 von 24 Paaren" in text and "Deutung" in text


def test_view_verdict_B_does_not_claim_the_delay_explanation():
    kind, text = ev.verdict_view("B", 4.12, 24, 24)
    assert kind == "warning" and "+4,12 %" in text and "24 von 24" not in text


def test_view_verdict_C_is_success_at_zero_and_warning_otherwise():
    assert ev.verdict_view("C", 1e-9, 24, 24)[0] == "success"
    assert "0,00 %" in ev.verdict_view("C", 0.0, 24, 24)[1]
    kind, text = ev.verdict_view("C", 0.5, 24, 24)
    assert kind == "warning" and "0,50 %" in text


def test_view_verdict_threshold_for_A_and_B_is_0_005_percent():
    assert ev.verdict_view("A", 0.004, 1, 1)[0] == "info"
    assert ev.verdict_view("A", 0.006, 1, 1)[0] == "warning"


def test_levels_verdict_without_offset_names_the_levels_and_with_offset_the_extra_cost():
    assert ev.verdict_levels(0, 0, 10, 13, 6.6, 6.6) == ("info", "Niveaus der Zerlegung: S1 = 10, S2 = 13.")
    kind, text = ev.verdict_levels(0, 3, 10, 16, 6.732, 6.6)             # +2,0 %
    assert kind == "info" and "S1 = 10 und S2 = 16" in text and "+2,00 %" in text and "6,73200" in text


def test_dp_live_verdict_success_below_1e6_and_warning_above():
    kind, text = ev.verdict_dp_live(6.6012936, 6.6012936, 8.24, 197568)
    assert kind == "success" and "197.568 Zuständen" in text and "8,2 s" in text and "6,60129" in text
    kind, text = ev.verdict_dp_live(6.6013, 6.6012936, 0.04, 760)
    assert kind == "warning" and "weicht" in text


def test_trend_verdict_counts_and_names_the_exception():
    kind, text = ev.verdict_trend("A", 12, 12, 15, 16, [(2, 1, 0.8)])
    assert kind == "info" and "12 von 12 Reihen" in text and "15 von 16 Reihen" in text and "Ausnahme: Lieferzeit 2/1, h_w 0,8." in text
    assert "Ausnahme" not in ev.verdict_trend("A", 12, 12, 16, 16, [])[1]


def test_delay_verdict_counts_equal_pairs_and_reports_the_largest_difference():
    pairs = [dict(diff=1e-13)] * 23 + [dict(diff=3e-4)]
    kind, text = ev.verdict_delay(pairs)
    assert kind == "warning" and "23 von 24 Paaren" in text and "3.0e-04" in text
    assert "Deutung" in text and "in 24 Zellen dieses Modells" in text  # Wirkung wie kürzere Lieferzeit ist Deutung mit Gültigkeitsbereich
    assert ev.verdict_delay([dict(diff=1e-13)] * 24)[0] == "success"


def test_scaling_verdict_names_first_and_last_row():
    kind, text = ev.verdict_scaling(SC.table())
    assert kind == "info" and "760 Zustände bei Lieferzeit 1/1" in text and "11.437.365.184 bei 4/4" in text and "133.940 Rechenschritte" in text and "1.218.854 bei 4/4" in text


def test_lag_verdict_names_best_lag_for_both_rules():
    kind, text = ev.verdict_lag("A", {1: 1.0, 2: 0.17, 3: 0.1}, {1: 0.17, 2: 0.848, 3: 0.1})
    assert kind == "info" and "100,0 %" in text and "t - 1" in text and "84,8 %" in text and "t - 2" in text and "Sicht A" in text
