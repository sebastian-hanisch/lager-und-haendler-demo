"""Vorgerechnete Ergebnisse: die 48 Zellen aus data/lah_grid.json (erzeugt von tools/build_grid.py) und die Zählungen darüber.

Die Regler der App liegen ausschließlich auf gemessenen Zellen; es gibt keine Nächste-Zelle-Regel. Alle Zählungen laufen über die übergebenen Zellen (Standard:
alle 48), damit Tests sie an handgesetzten Zeilen prüfen können."""

import json
from functools import lru_cache
from pathlib import Path

import lah_constants as C

GRID_PATH = Path(__file__).resolve().parent / "data" / "lah_grid.json"


@lru_cache(maxsize=1)
def load_grid() -> tuple:
    payload = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    return tuple(payload["cells"])


def get_cell(lr: int, lw: int, h_w: float, p: float, cells=None) -> dict:
    """Die gemessene Zelle; ein Wert außerhalb des Rasters ist ein Fehler (keine Nächste-Zelle-Regel)."""
    for c in cells if cells is not None else load_grid():
        if c["Lr"] == lr and c["Lw"] == lw and abs(c["h_w"] - h_w) < 1e-9 and abs(c["p"] - p) < 1e-9:
            return c
    raise KeyError(f"keine gemessene Zelle für Lieferzeit {lr}/{lw}, h_w {h_w}, p {p}")


def gap_percent(g: float, g_dp: float) -> float:
    """Lücke einer Regel zur gemeinsamen DP in Prozent der DP-Kosten (positiv = teurer als das Optimum)."""
    return 100.0 * (g / g_dp - 1.0)


def rule_gap(cell: dict, kind: str) -> float:
    return gap_percent(cell["rules"][kind]["g"], cell["g_dp"])


def gap_range(kind: str, cells=None) -> tuple:
    gaps = [rule_gap(c, kind) for c in (cells if cells is not None else load_grid())]
    return min(gaps), max(gaps)


def _series(cells, fixed: tuple, vary: str):
    """Zellen, die in den Schlüsseln `fixed` übereinstimmen, entlang `vary` sortiert, gruppiert."""
    groups = {}
    for c in cells:
        groups.setdefault(tuple(c[k] for k in fixed), []).append(c)
    return {k: sorted(v, key=lambda c: c[vary]) for k, v in groups.items()}


def monotone_in_hw(kind: str, cells=None) -> tuple:
    """(Reihen, in denen die Lücke mit h_w strikt wächst, Zahl der Reihen); Reihe = feste Lieferzeiten und p."""
    series = _series(cells if cells is not None else load_grid(), ("Lr", "Lw", "p"), "h_w")
    ok = sum(all(rule_gap(b, kind) > rule_gap(a, kind) for a, b in zip(s, s[1:])) for s in series.values())
    return ok, len(series)


def monotone_in_p(kind: str, cells=None) -> tuple:
    """(Reihen, in denen die Lücke mit p strikt fällt, Zahl der Reihen, Liste der Ausnahmen); Reihe = feste Lieferzeiten und h_w."""
    series = _series(cells if cells is not None else load_grid(), ("Lr", "Lw", "h_w"), "p")
    bad = [k for k, s in series.items() if not all(rule_gap(b, kind) < rule_gap(a, kind) for a, b in zip(s, s[1:]))]
    return len(series) - len(bad), len(series), sorted(bad)


def lead_effect(kind: str, which: str, cells=None) -> tuple:
    """Vergleich der Lücke bei Lieferzeit 2 gegen 1 an derselben Stelle: (Paare mit kleinerer Lücke bei 2, Zahl der Paare, Ausnahmen).

    which = 'Lr' vergleicht Lr = 2 gegen Lr = 1 (bei gleichem Lw, h_w, p), which = 'Lw' entsprechend."""
    cells = list(cells if cells is not None else load_grid())
    other = "Lw" if which == "Lr" else "Lr"
    index = {(c[which], c[other], c["h_w"], c["p"]): c for c in cells}
    pairs, bad = 0, []
    for (lead, oth, h_w, p), c in sorted(index.items()):
        if lead != 1:
            continue
        pairs += 1
        two = index[(2, oth, h_w, p)]
        if not rule_gap(two, kind) < rule_gap(c, kind):
            bad.append(dict(Lr=c["Lr"] if which == "Lw" else 1, Lw=c["Lw"] if which == "Lr" else 1, h_w=h_w, p=p, gap1=rule_gap(c, kind), gap2=rule_gap(two, kind)))
    return pairs - len(bad), pairs, bad


def delay_pairs(cells=None) -> list:
    """Paare 'A bei Lw = 1' gegen 'B bei Lw = 2' (gleiches Lr, h_w, p): Kosten der besten Niveaus je Regel und Differenz."""
    cells = list(cells if cells is not None else load_grid())
    index = {(c["Lr"], c["Lw"], c["h_w"], c["p"]): c for c in cells}
    out = []
    for (lr, lw, h_w, p), c in sorted(index.items()):
        if lw != 1:
            continue
        b2 = index[(lr, 2, h_w, p)]
        ga, gb = c["rules"]["A"]["g"], b2["rules"]["B"]["g"]
        out.append(dict(Lr=lr, h_w=h_w, p=p, g_A=ga, g_B_next=gb, diff=abs(ga - gb)))
    return out


def s1_independent_of_lw(cells=None) -> tuple:
    """(Paare mit gleichem S1 bei Lw = 1 und 2, Zahl der Paare)."""
    cells = list(cells if cells is not None else load_grid())
    index = {(c["Lr"], c["Lw"], c["h_w"], c["p"]): c for c in cells}
    pairs = [(index[(lr, 1, h, p)], index[(lr, 2, h, p)]) for (lr, lw, h, p) in index if lw == 1]
    return sum(a["S1"] == b["S1"] for a, b in pairs), len(pairs)


def s1_grows_with_lr(cells=None) -> tuple:
    cells = list(cells if cells is not None else load_grid())
    index = {(c["Lr"], c["Lw"], c["h_w"], c["p"]): c for c in cells}
    pairs = [(index[(1, lw, h, p)], index[(2, lw, h, p)]) for (lr, lw, h, p) in index if lr == 1]
    return sum(b["S1"] > a["S1"] for a, b in pairs), len(pairs)


def spread_grows_with_lw(cells=None) -> tuple:
    """S2 - S1 wächst mit Lw (strikt), je Paar Lw = 1 gegen 2 bei gleichem Lr, h_w, p."""
    cells = list(cells if cells is not None else load_grid())
    index = {(c["Lr"], c["Lw"], c["h_w"], c["p"]): c for c in cells}
    pairs = [(index[(lr, 1, h, p)], index[(lr, 2, h, p)]) for (lr, lw, h, p) in index if lw == 1]
    return sum((b["S2"] - b["S1"]) > (a["S2"] - a["S1"]) for a, b in pairs), len(pairs)


def echelon_equals_dp(cells=None, tol: float = 1e-5) -> dict:
    """Zählungen zur Kernaussage 'Zerlegung = gemeinsame DP' über die Zellen."""
    cells = list(cells if cells is not None else load_grid())
    n = len(cells)
    return dict(
        n=n,
        cost_equal=sum(abs(c["g_echelon"] - c["g_dp"]) < tol for c in cells),
        max_cost_diff=max(abs(c["g_echelon"] - c["g_dp"]) for c in cells),
        best_enum_same_cost=sum(abs(c["echelon_best"] - c["g_echelon"]) < 1e-9 for c in cells),
        best_enum_same_levels=sum(tuple(c["echelon_best_par"]) == (c["S1"], c["S2"]) for c in cells),
        formula_same_levels=sum(tuple(c["formula_best_par"]) == (c["S1"], c["S2"]) for c in cells),
        max_action_diff=max(c["diff_mass"] for c in cells),
        max_cap=max(c["cap_dp"] for c in cells),
        max_edge=max(max(c["edge_x_low"], c["edge_x_high"], c["edge_w"]) for c in cells),
        c_equals_dp=sum(abs(c["rules"]["C"]["g"] - c["g_dp"]) < 1e-6 * c["g_dp"] for c in cells),
        a_gt_b_gt_0=sum(c["rules"]["A"]["g"] > c["rules"]["B"]["g"] > c["g_dp"] * (1 + 1e-9) for c in cells),
    )


def warehouse_stock_larger(cells=None) -> tuple:
    """Zellen, in denen die lokale Regel A im Mittel mehr Lagerbestand hält als die Echelon-Politik (Deutung 'Mehrbestand als Ersatz für fehlende Sicht')."""
    cells = list(cells if cells is not None else load_grid())
    return sum(c["rules"]["A"]["mean_w"] > c["rules"]["echelon"]["mean_w"] for c in cells), len(cells)


def cell_rows(cells=None) -> list:
    """Tabellenzeilen für die App: je Zelle Lieferzeiten, h_w, p, Niveaus und die Lücken der drei Sichten (Prozent)."""
    rows = []
    for c in cells if cells is not None else load_grid():
        rows.append(dict(Lr=c["Lr"], Lw=c["Lw"], h_w=c["h_w"], p=c["p"], S1=c["S1"], S2=c["S2"], g_dp=c["g_dp"],
                         gap_A=rule_gap(c, "A"), gap_B=rule_gap(c, "B"), gap_C=rule_gap(c, "C")))
    return rows


def preset_cells() -> list:
    return [get_cell(v["lr"], v["lw"], v["hw"], v["p"]) for v in C.PRESETS.values()]
