"""Die vorgerechnete Datei data/lah_grid.json: vollständig, klein, und stichprobenweise gegen eine Neuberechnung mit tools/build_grid.py."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import lah_constants as C  # noqa: E402
import lah_results as R  # noqa: E402


def test_grid_has_all_48_cells_exactly_once():
    cells = R.load_grid()
    keys = {(c["Lr"], c["Lw"], c["h_w"], c["p"]) for c in cells}
    assert len(cells) == C.N_CELLS == 48 == len(keys)
    assert keys == {(lr, lw, h, float(p)) for lr in C.LEADS for lw in C.LEADS for h in C.H_W_GRID for p in C.P_GRID}


def test_grid_header_matches_the_model_constants():
    payload = json.loads(R.GRID_PATH.read_text(encoding="utf-8"))
    assert (payload["mean"], payload["h_r"], payload["Q"]) == (C.MEAN, C.H_R, C.Q_MAX) and payload["version"] == 1


def test_repository_stays_below_3_megabytes():
    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file() and ".git" not in f.parts and "__pycache__" not in f.parts and ".pytest_cache" not in f.parts)
    assert total < 3_000_000, total
    assert R.GRID_PATH.stat().st_size < 200_000


def test_every_cell_has_all_four_rules_with_cost_parts_that_sum_up():
    for c in R.load_grid():
        assert set(c["rules"]) == {"echelon", "A", "B", "C"}
        for rec in c["rules"].values():
            assert rec["hold_w"] + rec["hold_r"] + rec["short"] == pytest.approx(rec["g"], abs=1e-9)
            assert 0.0 <= rec["unfilled"] <= 1.0


def test_dp_never_undercuts_any_rule_and_the_dp_rasters_do_not_use_the_edge():
    """Eine DP, die ein bekanntes Optimum unterbietet, ist ein Randalarm (beide Ränder prüfen)."""
    for c in R.load_grid():
        assert all(c["rules"][k]["g"] >= c["g_dp"] - 1e-6 for k in c["rules"])
        assert c["edge_x_low"] < 1e-6 and c["edge_x_high"] < 1e-6 and c["edge_w"] < 1e-6


@pytest.mark.parametrize("lr,lw,h_w,p", [(1, 1, 0.4, 9.0), (1, 1, 0.8, 4.0), (1, 2, 0.2, 19.0)])
def test_cells_reproduce_from_the_build_script(lr, lw, h_w, p):
    import build_grid
    fresh = build_grid.cell(lr, lw, h_w, p, full_window=(lr, lw) == (1, 1))
    stored = R.get_cell(lr, lw, h_w, p)
    assert fresh["raster"] == stored["raster"] and (fresh["S1"], fresh["S2"]) == (stored["S1"], stored["S2"])
    for key in ("g_dp", "g_echelon", "g_formula"):
        assert fresh[key] == pytest.approx(stored[key], abs=1e-9)
    for kind in ("echelon", "A", "B", "C"):
        assert fresh["rules"][kind]["g"] == pytest.approx(stored["rules"][kind]["g"], abs=1e-9)
        assert (fresh["rules"][kind]["S1"], fresh["rules"][kind]["S2"]) == (stored["rules"][kind]["S1"], stored["rules"][kind]["S2"])


def test_hill_climbing_equals_the_full_window_in_lead_time_1_1():
    """Die Niveaus der lokalen Regeln aus dem Bergsteigen sind in Lieferzeit 1/1 gegen das volle Fenster geprüft: 12 Zellen mal zwei Regeln = 24 Vergleiche."""
    n = 0
    for c in R.load_grid():
        for kind in ("A", "B"):
            if "window_g" in c["rules"][kind]:
                assert c["rules"][kind]["window_g"] == pytest.approx(c["rules"][kind]["g"], abs=1e-9)
                n += 1
    assert n == 24
