"""PDF-Bericht der gewählten Zelle (fpdf2, Standardschrift: Umlaute sind darin erlaubt, Sonderzeichen wie Pfeile nicht)."""

from fpdf import FPDF

import lah_constants as C
import lah_evaluation as ev
import lah_results as R


def _de(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def build_pdf(settings: ev.Settings, analysis: ev.Analysis) -> bytes:
    cell = analysis.cell
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, "Lager und Händler: Zerlegung statt gemeinsamer Rechnung", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 5,
        f"Zelle: Lieferzeit Händler/Lager {settings.lr}/{settings.lw}, Lagerkostensatz h_w {_de(settings.hw, 1)} (Händler 1,0), Fehlmengenkosten p {settings.p}. "
        f"Modell: Poisson-Nachfrage, Mittel 3, Rückstand beim Händler, Bestellkosten 0, Bestellgrenze {C.Q_MAX} je Stufe. Alle Kosten je Periode.",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Zerlegung", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, f"S1 = {cell['S1']}, S2 = {cell['S2']}; gemeinsame DP {_de(cell['g_dp'], 5)}; Echelon-Politik {_de(cell['g_echelon'], 5)}.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Regeln im Vergleich", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    names = {"echelon": "Echelon (Zerlegung)", "A": "lokal, Sicht A", "B": "lokal, Sicht B", "C": "lokal, Sicht C"}
    for kind, name in names.items():
        rec = cell["rules"][kind]
        gap = "0,00 %" if kind == "echelon" else _de(R.rule_gap(cell, kind), 2).join(["+", " %"])
        pdf.cell(60, 6, name)
        pdf.cell(35, 6, _de(rec["g"], 5))
        pdf.cell(30, 6, gap)
        pdf.cell(0, 6, f"Niveaus ({rec['S1']}, {rec['S2']})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0, 4.5,
        "Lücke = Kosten der Regel geteilt durch die Kosten der gemeinsamen DP, minus 1. Bei den lokalen Regeln sind die Niveaus per Bergsteigen gesucht (nicht als globales "
        "Optimum bewiesen). Sicht A: das Lager kennt nur seinen Bestand; B: es sieht, was es liefert; C: es sieht die volle Bestellung des Händlers.",
        new_x="LMARGIN", new_y="NEXT",
    )
    return bytes(pdf.output())
