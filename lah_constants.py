"""Konstanten: Regler-Grenzen, gemessene Zellen, feste Annahmen, Farben und Presets."""

# --- Modell (feste Annahmen, im Text der App und in der README genannt) ------------------------------------------------------------------
MEAN = 3.0                 # mittlere Nachfrage je Periode (Poisson)
H_R = 1.0                  # Lagerkostensatz für den Bestand des Händlers (insgesamt)
Q_MAX = 13                 # Bestellgrenze je Stufe; Randmasse (Anteil der Zeit an der Grenze) höchstens 3,2e-5 in allen 48 Zellen
DMAX = 18                  # Nachfrage wird bei 18 gedeckelt (Rest der Poisson-Verteilung auf 18)

# --- gemessene Zellen: die Regler liegen ausschließlich auf diesen Werten (keine Nächste-Zelle-Regel) -------------------------------------
LEADS = (1, 2)             # Lieferzeit des Händlers beim Lager und des Lagers beim Lieferanten, je Stufe
H_W_GRID = (0.2, 0.4, 0.6, 0.8)
P_GRID = (4, 9, 19)
N_CELLS = len(LEADS) * len(LEADS) * len(H_W_GRID) * len(P_GRID)    # 48

DEFAULT_LR, DEFAULT_LW, DEFAULT_HW, DEFAULT_P = 1, 1, 0.4, 9
VIEWS = ("A", "B", "C")
DEFAULT_VIEW = "A"
VIEW_LABEL = {
    "A": "A: nur der eigene Bestand",
    "B": "B: was das Lager liefert",
    "C": "C: die volle Bestellung",
}
VIEW_HELP = (
    "A: das Lager entscheidet gleichzeitig mit dem Händler und kennt nur seinen eigenen Bestand (die lokale Regel). "
    "B: das Lager sieht, was es in dieser Periode an den Händler liefert. "
    "C: das Lager sieht die volle Bestellung des Händlers, auch den Teil, den es nicht liefern kann."
)

D1_MIN, D1_MAX = -4, 4      # Abweichung von S1 der Zerlegung (nur Echelon-Politik, nur geschlossene Formel)
D2_MIN, D2_MAX = -6, 6
SEED_MAX, DEFAULT_SEED = 999999, 11
SHOW_MIN, SHOW_MAX, DEFAULT_SHOW = 20, 60, 30     # Zahl der gezeigten Perioden im Verlauf
RUN_LENGTH = 20000         # Länge des Nachfragestroms für die Bestellquoten (Perioden nach der Einschwingzeit)
BURN_IN = 2000

DP_AUTO_MAX_SUM = 3        # Lr + Lw bis hierhin rechnet die App die gemeinsame DP selbst; bei 2/2 auf Knopfdruck

# --- Handbeispiel (von Hand rechenbar; fest, unabhängig von den Reglern) ------------------------------------------------------------------
EXAMPLE = dict(h_w=0.5, h_r=1.0, p=3.0, Lr=1, Lw=1, Q=4, W=6, xmin=-8, xmax=4, dist=(0.25, 0.5, 0.25))

# --- Farben (Okabe-Ito, auch für Rot-Grün-Schwäche unterscheidbar) --------------------------------------------------------------------
COLOR_ECHELON = "#0072B2"
COLOR_A = "#D55E00"
COLOR_B = "#E69F00"
COLOR_C = "#009E73"
COLOR_DEMAND = "rgba(120,120,120,0.55)"
COLOR_HOLD_W = "#56B4E9"
COLOR_HOLD_R = "#0072B2"
COLOR_SHORT = "#D55E00"
VIEW_COLOR = {"A": COLOR_A, "B": COLOR_B, "C": COLOR_C}

PRESETS = {
    "Hauptfall (1/1, h_w 0,4, p 9)": {"lr": 1, "lw": 1, "hw": 0.4, "p": 9, "view": "A", "d1": 0, "d2": 0},
    "Langer Weg zum Lager (1/2)": {"lr": 1, "lw": 2, "hw": 0.4, "p": 9, "view": "A", "d1": 0, "d2": 0},
    "Langer Weg zum Händler (2/1)": {"lr": 2, "lw": 1, "hw": 0.4, "p": 9, "view": "A", "d1": 0, "d2": 0},
    "Teures Lager (h_w 0,8)": {"lr": 1, "lw": 1, "hw": 0.8, "p": 9, "view": "A", "d1": 0, "d2": 0},
    "Billiges Lager (h_w 0,2)": {"lr": 1, "lw": 1, "hw": 0.2, "p": 9, "view": "A", "d1": 0, "d2": 0},
    "Hohe Fehlmengenkosten (p 19)": {"lr": 1, "lw": 1, "hw": 0.4, "p": 19, "view": "A", "d1": 0, "d2": 0},
}
PRESET_HELP = {
    "Hauptfall (1/1, h_w 0,4, p 9)": "Lieferzeit 1 beim Händler und 1 beim Lager, Lagerkostensatz 0,4 gegen 1,0 beim Händler, Fehlmenge 9; Sicht A, Niveaus der Zerlegung. "
                                     "Die Zahl der lokalen Regel (+11,5 %) stammt aus diesem Fall.",
    "Langer Weg zum Lager (1/2)": "Das Lager wartet zwei Perioden auf den Lieferanten; das Niveau des Lagers steigt, das des Händlers nicht.",
    "Langer Weg zum Händler (2/1)": "Der Händler wartet zwei Perioden auf das Lager; beide Niveaus steigen.",
    "Teures Lager (h_w 0,8)": "Lagerhaltung im Lager fast so teuer wie beim Händler: das Lager hält nur wenig mehr, als durchläuft; die Lücke der lokalen Regel wächst.",
    "Billiges Lager (h_w 0,2)": "Lagerhaltung im Lager billig: ein Mehrbestand im Lager schadet wenig, die Lücke der lokalen Regel ist am kleinsten.",
    "Hohe Fehlmengenkosten (p 19)": "Fehlmenge kostet 19 je Stück: die Niveaus steigen, die Lücke der lokalen Regel fällt meist (Ausnahme in der README genannt).",
}
