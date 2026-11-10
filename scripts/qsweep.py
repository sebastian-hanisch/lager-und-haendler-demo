"""Bestellgrenze Q: gilt Echelon = gemeinsame DP auch mit einer Grenze, die oft greift? Hauptzelle Lieferzeit 1/1, h_w = 0,4, p = 9, Poisson Mittel 3.

Je Q: gemeinsame DP, beste Echelon-Politik (Aufzählung S1 = 1..16, S2 = S1..S1+16 mit exakter Auswertung auf der Kette), Zeitanteil an der Grenze unter der DP-Politik,
Randmassen, und die Lücke der Echelon-Politik mit den Niveaus der Zerlegung (ohne Grenze gerechnet).
Aufruf (Repo-Wurzel): python scripts/qsweep.py   (rund eine Minute; OMP_NUM_THREADS=1)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lah_chain as CH  # noqa: E402
import lah_decompose as D  # noqa: E402
import lah_dp  # noqa: E402
import lah_policies as P  # noqa: E402
from lah_model import Params, StateSpace  # noqa: E402

QS = (3, 4, 5, 6, 7, 8, 10, 13)


def measure(Q: int) -> dict:
    zer = D.solve(Params())
    space = StateSpace(Params(Q=Q, xmin=-26, xmax=12, W=26))
    dp = lah_dp.value_iteration(space)
    res = CH.evaluate(space, dp.policy)
    best = (1e9, None)
    for s1 in range(1, 17):
        for s2 in range(s1, s1 + 17):
            g = CH.evaluate(space, P.policy_actions(space, "echelon", s1, s2)).g
            if g < best[0] - 1e-12:
                best = (g, (s1, s2))
    g_zer = CH.evaluate(space, P.policy_actions(space, "echelon", zer.S1, zer.S2)).g
    return dict(Q=Q, states=space.n_states, g_dp=dp.g, best_par=best[1], g_best=best[0], gap_best=100 * (best[0] / dp.g - 1), cap=res.cap,
                edge=max(res.edge_x_low, res.edge_x_high, res.edge_w), gap_zerlegung=100 * (g_zer / dp.g - 1))


if __name__ == "__main__":
    for Q in QS:
        m = measure(Q)
        print(f"Q {m['Q']:2d} | g_DP {m['g_dp']:.5f} | beste Echelon {m['best_par']} {m['g_best']:.5f} | Lücke {m['gap_best']:+.3f} % | Zeitanteil an der Grenze {m['cap']:.1e} | "
              f"Randmasse {m['edge']:.0e} | Niveaus der Zerlegung (10, 13): {m['gap_zerlegung']:+.3f} %", flush=True)
