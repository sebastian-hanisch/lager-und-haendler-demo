"""Erzeugt data/lah_grid.json: die 48 gemessenen Zellen (Lieferzeit Händler/Lager 1 oder 2, h_w 0,2 bis 0,8, p 4, 9, 19; Poisson Mittel 3, h_r = 1, Bestellgrenze 13).

Je Zelle: Zerlegung (geschlossene Formel), gemeinsame DP (Wertiteration, exakt), beste Echelon-Politik aus der Aufzählung (Kette, Fenster um die Zerlegung und
weites Fenster mit der Formel), die drei lokalen Regeln A, B, C (Niveaus per Bergsteigen aus drei Startpunkten, in 1/1 gegen das volle Fenster geprüft) mit exakten
Kosten, Kostenanteilen und Randmassen. Das Raster wächst, bis die DP-Politik weder unten noch oben noch im Lager den Rand nutzt (Randmasse unter 1e-6).

Aufruf (Repo-Wurzel; OMP_NUM_THREADS=1; höchstens zwei Prozesse zugleich):
    python tools/build_grid.py --leads 1-1,1-2,2-1 --out scratch/teil_a.json
    python tools/build_grid.py --leads 2-2 --out scratch/teil_b.json
    python tools/build_grid.py --merge scratch/teil_a.json scratch/teil_b.json --out data/lah_grid.json
Die Laufzeit liegt bei einigen Minuten (2/2 dominiert: je Zelle rund eine halbe Minute)."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lah_chain as CH  # noqa: E402
import lah_constants as C  # noqa: E402
import lah_decompose as D  # noqa: E402
import lah_policies as P  # noqa: E402
from lah_algorithm import fit_raster  # noqa: E402
from lah_model import Params, StateSpace  # noqa: E402

START_STEPS = (0, 3, 6)     # Startpunkte des Bergsteigens: Sw0 = S2 - S1 + k
GRID_VERSION = 1


def climb(evaluate, start: tuple) -> tuple:
    """Bergsteigen auf (S1, Sw) mit den acht Nachbarn; stoppt, wenn kein Nachbar um mehr als 1e-12 besser ist."""
    cur = start
    while True:
        best = cur
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                cand = (cur[0] + d1, cur[1] + d2)
                if cand[0] < 1 or cand[1] < 0:
                    continue
                if evaluate(cand) < evaluate(best) - 1e-12:
                    best = cand
        if best == cur:
            return cur
        cur = best


def rule_record(space: StateSpace, kind: str, S1: int, S2: int) -> dict:
    res = CH.evaluate(space, P.policy_actions(space, kind, S1, S2))
    return dict(
        g=res.g, S1=S1, S2=S2, unfilled=CH.unfilled_share(space, res.pi, S1), cap=res.cap, edge_w=res.edge_w, edge_x_low=res.edge_x_low,
        n_reachable=res.n_reachable, **res.parts,
    )


def cell(Lr: int, Lw: int, h_w: float, p: float, full_window: bool = False) -> dict:
    base = Params(Lr=Lr, Lw=Lw, h_w=h_w, p=p, Q=C.Q_MAX)
    joint = fit_raster(Lr, Lw, h_w, p)
    dec, space, dp, dp_chain = joint.decomposition, joint.space, joint.dp, joint.dp_chain
    S1, S2 = dec.S1, dec.S2
    xmin, xmax, W = joint.params.xmin, joint.params.xmax, joint.params.W
    pe = P.policy_actions(space, "echelon", S1, S2)
    ech = CH.evaluate(space, pe)
    diff_mass = float(dp_chain.pi[(dp.policy != pe) & (dp_chain.pi > 1e-12)].sum())
    best = (1e9, None)
    for a in range(max(1, S1 - 2), S1 + 3):
        for b in range(max(a, S2 - 3), S2 + 4):
            gg = CH.evaluate(space, P.policy_actions(space, "echelon", a, b)).g
            if gg < best[0] - 1e-12:
                best = (gg, (a, b))
    formula_best = (1e9, None)
    for a in range(1, S1 + 9):
        for b in range(a, a + 27):
            gg = D.echelon_cost(base, a, b)
            if gg < formula_best[0] - 1e-12:
                formula_best = (gg, (a, b))
    out = dict(
        Lr=Lr, Lw=Lw, h_w=h_w, p=p, raster=dict(xmin=xmin, xmax=xmax, W=W), n_states=space.n_states, S1=S1, S2=S2, g_formula=dec.g, ties=[dec.ties1, dec.ties2],
        g_dp=dp.g, g_dp_chain=dp_chain.g, dp_iterations=dp.iterations, g_echelon=ech.g, echelon_best=best[0], echelon_best_par=list(best[1]),
        formula_best=formula_best[0], formula_best_par=list(formula_best[1]), diff_mass=diff_mass,
        edge_x_low=dp_chain.edge_x_low, edge_x_high=dp_chain.edge_x_high, edge_w=dp_chain.edge_w, cap_dp=dp_chain.cap, n_reachable=dp_chain.n_reachable,
    )
    rules = {"echelon": rule_record(space, "echelon", S1, S2)}
    for kind in ("A", "B"):
        cache = {}

        def evaluate(c, kind=kind, cache=cache):
            if c not in cache:
                cache[c] = CH.evaluate(space, P.policy_actions(space, kind, *c)).g
            return cache[c]

        starts = [climb(evaluate, (S1, max(S2 - S1, 0) + k)) for k in START_STEPS]
        chosen = min(starts, key=evaluate)
        rules[kind] = rule_record(space, kind, *chosen)
        rules[kind]["n_evaluations"] = len(cache)
        if full_window:
            win = (1e9, None)
            for a in range(max(1, S1 - 3), S1 + 4):
                for b in range(0, 21):
                    gg = CH.evaluate(space, P.policy_actions(space, kind, a, b)).g
                    if gg < win[0] - 1e-12:
                        win = (gg, (a, b))
            rules[kind]["window_g"] = win[0]
            rules[kind]["window_par"] = list(win[1])
    rules["C"] = rule_record(space, "C", S1, S2 - S1)
    out["rules"] = rules
    return out


def build(leads, part: Path | None = None) -> list[dict]:
    """Rechnet die Zellen; mit `part` wird nach jeder Zelle gesichert und ein abgebrochener Lauf setzt dort fort."""
    cells = json.loads(part.read_text(encoding="utf-8")) if part is not None and part.exists() else []
    done = {(c["Lr"], c["Lw"], c["p"], c["h_w"]) for c in cells}
    for Lr, Lw in leads:
        for p in C.P_GRID:
            for h_w in C.H_W_GRID:
                if (Lr, Lw, float(p), h_w) in done:
                    continue
                t0 = time.time()
                cells.append(cell(Lr, Lw, h_w, float(p), full_window=(Lr, Lw) == (1, 1)))
                if part is not None:
                    part.write_text(json.dumps(cells, separators=(",", ":")), encoding="utf-8", newline="\n")
                print(f"{Lr}/{Lw} h_w {h_w} p {p}: {time.time() - t0:.0f} s", flush=True)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leads", default="1-1,1-2,2-1,2-2", help="Lieferzeit-Kombinationen Händler-Lager, z. B. 1-1,1-2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", nargs="*", help="statt zu rechnen: diese Teildateien zusammenführen")
    args = ap.parse_args()
    if args.merge:
        cells = [c for f in args.merge for c in json.loads(Path(f).read_text(encoding="utf-8"))["cells"]]
    else:
        leads = [tuple(int(v) for v in s.split("-")) for s in args.leads.split(",")]
        cells = build(leads, Path(args.out).with_suffix(".part"))
    cells.sort(key=lambda c: (c["Lr"], c["Lw"], c["p"], c["h_w"]))
    payload = dict(version=GRID_VERSION, mean=C.MEAN, h_r=C.H_R, Q=C.Q_MAX, cells=cells)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(cells)} Zellen nach {path} ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
