# Lager und Händler: Zerlegung statt gemeinsamer Rechnung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lager-und-haendler-demo.streamlit.app/)**

Stück der **bestandspolitiken** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vorgänger ist das Stück zur Lieferzeit (die Bestandsposition als eine Zahl: Kante `fix`, denn **eine** Bestandsposition genügt als Zustand nur für **einen** Bestand); Nachfolger ist die Frage, wo die exakte Rechnung aufgibt. Das Vehikel ist dasselbe wie in der ganzen Reihe: ein Artikel, eine Entscheidung je Periode („wie viel bestellen“), Poisson-Nachfrage mit Mittel 3, Fehlmenge als Rückstand. Neu ist der zweite Bestand: ein **Lager** beliefert einen **Händler**, beide bestellen.

## Warum dieses Problem

Mit zwei gekoppelten Beständen wächst der Zustandsraum der gemeinsamen Rückwärtsrechnung mit jeder zusätzlichen Periode Lieferzeit auf das 14,6- bis 16,9-Fache (bei 14 Werten je Pipelineplatz, Raster der Rechenlast-Tafel). Die **Kernfrage**: **Reicht es, jede Stufe für sich zu steuern?** Die Antwort hat zwei Teile. Erstens: Die optimale Politik zweier Stufen in Reihe ist im Modell der Demo (Bestellgrenze 13) ein Base-Stock auf dem **Echelon-Bestand** (Bestand der Stufe plus alles, was hinter ihr liegt), und ihre zwei Niveaus S1, S2 folgen aus zwei eindimensionalen Rechnungen, die nacheinander gekoppelt sind (Zerlegung nach Clark und Scarf). Zweitens: Was „jede Stufe für sich“ kostet, hängt davon ab, **was das Lager vom Händler sieht**. Eine Aussage wie „die lokale Regel verliert 9 bis 11 %“ gilt nur für Sicht A (bei `h_w` 0,4 und p 9: +11,51 / +9,88 / +8,93 % für Lieferzeit 1/1, 1/2, 2/1); mit Sicht B sind es bei denselben Einstellungen +3,3 bis +4,1 % (vier Lieferzeit-Kombinationen), mit Sicht C nichts (Zahlen und Voraussetzungen unten).

Namensabgrenzung: In dieser Demo heißt es „zwei Stufen in Reihe“ oder „Lager und Händler“; das Wort „zweistufig“ ist in diesem Portfolio für „erst vorhersagen, dann optimieren“ vergeben.

## Modell

Lieferant (unbegrenzter Vorrat) → **Lager** → **Händler** → Kunden. Zu Beginn jeder Periode bestellt der Händler `q_r ≤ w` beim Lager (`w` = Lagerbestand), das Lager `q_w` beim Lieferanten; die Bestellungen kommen nach `L_r` bzw. `L_w` Perioden an (bei Lieferzeit 1 am Ende der laufenden Periode, nach der Nachfrage). Die Nachfrage D je Periode ist Poisson mit Mittel 3 (bei 18 gedeckelt: der Rest der Verteilung liegt auf 18). Unbediente Nachfrage beim Händler wird nachgeliefert (Rückstand); beim Lager gibt es keinen Rückstand, der Händler bekommt höchstens, was das Lager hat. Kosten je Periode: `h_w · (Lagerbestand + Ware auf dem Weg zum Händler) + h_r · (Händlerbestand − D)⁺ + p · (D − Händlerbestand)⁺` mit `h_r = 1` (Händlerbestand insgesamt), `h_w ∈ {0,2; 0,4; 0,6; 0,8}` und `p ∈ {4, 9, 19}`; Bestellkosten 0; Bestellgrenze **13** je Stufe. Lieferzeit Händler/Lager (`L_r/L_w`): 1 oder 2 je Stufe (4 Kombinationen, 12 Kostenpaare je Kombination: **48 Zellen**). Die Regler der App liegen ausschließlich auf diesen gemessenen Zellen; es gibt keine Nächste-Zelle-Regel.

**Zerlegung.** Echelon-Sätze: Händlerstufe `h_r − h_w` je Stück Händlerbestand und `p + h_w` je fehlendes Stück, Lagerstufe `h_w` je Stück Echelon-Bestand. Schritt 1 minimiert die Kosten `L1(y)` der Händlerstufe über die Nachfrage von `L_r + 1` Perioden (`S1` = Minimum). Schritt 2 minimiert die Kosten `G2(y)` der Lagerstufe mit der von der Händlerstufe induzierten Kostenfunktion, in der der Echelon-Bestand der Händlerstufe höchstens `S1` ist (`S2` = Minimum, `g = G2(S2)` = mittlere Kosten je Periode der Politik). Die Politik: der Händler bestellt bis `S1` auf seiner Bestandsposition, das Lager bis `S2` auf dem Echelon-Bestand. Die vollständigen Formeln stehen im Expander „📐 Mathematische Formulierung“ der App. Die Formeln für `L1` und `G2` sind **eigene Herleitung** für das Zeitschema dieser Demo (Kosten auf dem Bestand zu Beginn der Periode), numerisch gegen die gemeinsame Rechnung geprüft; die Struktur (erst Händlerstufe, dann Lager mit der induzierten Funktion) entspricht Chao und Zhou (2006), Abschnitt 2.1.

**Lokale Regel in drei Sichten.** Der Händler bestellt in allen Regeln bis `S1` auf seiner Bestandsposition. Das Lager bestellt bis `S_w` auf dem Bestand, den es sieht: **A** nur `w` plus Ware auf dem Weg zum Lager (das Lager entscheidet gleichzeitig mit dem Händler; das ist „die lokale Regel“ der Vorarbeit), **B** `w` minus das, was es in dieser Periode an den Händler liefert, **C** `w` minus die volle Bestellung des Händlers, auch den nicht lieferbaren Teil. Die Niveaus `(S1, S_w)` der Sichten A und B sind per Bergsteigen aus drei Startpunkten gesucht (in Lieferzeit 1/1 gegen das volle Fenster geprüft: 24 von 24 Vergleichen gleich; **kein Beweis** für das globale Optimum). Sicht C bekommt `S_w = S2 − S1` aus der Zerlegung.

## Methodik

- **Gemeinsame Rückwärtsrechnung** (relative Wertiteration) über den Zustand (Händlerbestand, Lagerbestand, Ware auf dem Weg zum Händler und zum Lager), Aktionen alle `(q_r, q_w)` mit `q_r ≤ w`. Der Erwartungswert über die Nachfrage läuft als Faltung entlang des Händlerbestands. Das Raster wächst, bis die DP-Politik weder unten noch oben noch im Lager den Rand nutzt (Randmasse unter 1e-6; **beide Ränder** werden geprüft: bei zu engem Händlerraster geht Bestand bzw. Rückstand am Rand kostenlos verloren und die DP unterbietet scheinbar das Optimum; ein zu enges Lagerraster macht sie zu teuer).
- **Exakte Auswertung** jeder Regel auf der Markov-Kette (stationäre Verteilung auf der erreichbaren Menge, direkt oder per Potenziteration der trägen Kette). Kosten, Kostenanteile, Lagerbestand und Randmassen sind exakte Erwartungswerte der stationären Verteilung, kein Zufall.
- **Simulation** nur für den Verlauf in der App und als Orakel: ganzzahliger Erzeuger (SplitMix64, Inversion der Verteilungsfunktion), unabhängig von der numpy-Version; Einschwingzeit 2000 Perioden, weil ein leerer Start die Kosten verfälscht.
- **Vorgerechnet** (`data/lah_grid.json`, erzeugt von `tools/build_grid.py`, rund 80 kB): je Zelle DP, Zerlegung, die vier Regeln mit Kosten, Niveaus, Kostenanteilen und Randmassen. **Live gerechnet:** Zerlegung, Verlauf, die gemeinsame DP bei Lieferzeit 1/1, 1/2 und 2/1 (jeweils unter einer Sekunde bis wenige Zehntelsekunden), bei 2/2 auf Knopfdruck (Größenordnung zehn Sekunden), das Handbeispiel.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für das Modell oben (Poisson-Nachfrage Mittel 3, Rückstand beim Händler, Bestellgrenze 13 je Stufe, `h_r = 1`, deterministische Lieferzeiten). Die **Lücke** einer Regel ist ihr Kostenüberschuss in Prozent der Kosten der gemeinsamen DP. „Zellen“ sind die 48 Zellen; die Zählungen laufen über alle.

| Frage | Befund | Test |
|---|---|---|
| Trifft die Zerlegung die gemeinsame DP? | In **48 von 48** Zellen sind die Kosten der Echelon-Politik mit den Niveaus der Zerlegung gleich denen der gemeinsamen DP (größter Unterschied unter 1e-8, gedruckt gleich auf fünf Stellen); die beste Echelon-Politik der Aufzählung hat in 48 von 48 Zellen dieselben Niveaus (Kette im Fenster S1 ± 2, S2 ± 3 und Formel im weiten Fenster). Die DP-Aktion weicht auf der DP-Verteilung höchstens in 3,4e-11 der Zeit von der Echelon-Aktion ab. | `test_claims.py::test_claim_decomposition_equals_joint_dp_in_48_of_48_cells` (exakt/Band) |
| Wie genau in der Hauptzelle (1/1, `h_w` 0,4, p 9)? | 6,6012936 je Periode (DP und Echelon, sieben Stellen); die Formel ohne Bestellgrenze gibt 6,6012914 (Unterschied 2,3e-6, Einfluss der Grenze). | `test_claim_main_cell_costs_to_seven_digits` |
| Randmassen | Zeitanteil an der Bestellgrenze höchstens 3,2e-5 (DP, alle Zellen); Händlerbestand unten höchstens 1,7e-9, Lagerbestand an der Rastergrenze 0; lokale Regeln: Bestellgrenze höchstens 5,9e-7, Händlerbestand unten bei Sicht A höchstens 1,2e-5 (alle unter 1e-4). | `test_claim_edge_masses_are_below_1e4` |
| Lokale Regel in der Hauptzelle | Sicht A **+11,51 %** (Niveaus 10, 9), Sicht B **+4,12 %** (10, 5), Sicht C **0,00 %** (10, 3). | `test_claim_three_views_in_the_main_cell` |
| … je Lieferzeit (`h_w` 0,4, p 9) | Sicht A: +11,51 / +9,88 / +8,93 / +7,78 % für 1/1, 1/2, 2/1, 2/2; Sicht B: +4,12 / +3,85 / +3,26 / +3,33 %; Sicht C überall 0,00 %. | `test_claim_gaps_by_lead_times_at_h_w_0_4_p_9` |
| … über alle 48 Zellen | Sicht A +3,73 bis +20,94 %, Sicht B +0,79 bis +11,27 %, Sicht C 0,00 % in 48 von 48; A > B > 0 in 48 von 48. | `test_claim_gap_ranges_over_all_48_cells` |
| Abhängigkeit von `h_w` | Die Lücke wächst mit `h_w` (0,2 → 0,8) in 12 von 12 Reihen (feste Lieferzeiten und p), für A und für B. Sicht A: +3,73 bis +6,95 % bei `h_w` 0,2, +11,50 bis +20,94 % bei 0,8. | `test_claim_trends_with_h_w_p_and_lead_times_with_named_exceptions`, `test_claim_gap_by_h_w_extremes` |
| Abhängigkeit von p | Sicht A: die Lücke fällt mit p (4 → 9 → 19) in 15 von 16 Reihen (feste Lieferzeiten und `h_w`); **Ausnahme** Lieferzeit 2/1, `h_w` 0,8: 13,35 % (p 9) → 13,83 % (p 19). Sicht B fällt nur in 12 von 16 Reihen (nicht gedeutet). | dieselben Tests |
| Abhängigkeit von den Lieferzeiten | Sicht A: bei Lieferzeit 2 des Händlers ist die Lücke kleiner als bei 1 in 24 von 24 Paaren, bei Lieferzeit 2 des Lagers kleiner als bei 1 in 23 von 24 (**Ausnahme** 1/1 → 1/2, `h_w` 0,8, p 4: 20,69 → 20,94 %). | dieselben Tests |
| Informationsverzug | Die Regel A bei Lagerlieferzeit 1 kostet so viel wie die Regel B bei Lagerlieferzeit 2 (beste Niveaus je Regel): gleich in **24 von 24** Paaren (größter Unterschied unter 1e-12); auch bei gleichen Niveaus gleich bis auf Rechengenauigkeit (relativ unter 1e-11). Ein Mehrwissen des Lagers um eine Periode wirkt wie eine um eine Periode kürzere Lagerlieferzeit (gemessen für Lagerlieferzeit 1 gegen 2 in 24 Zellen dieses Modells). Das ist **Deutung**, gestützt auf diese Gleichheit; als Ursache nicht bewiesen. | `test_claim_view_A_at_warehouse_lead_time_1_equals_view_B_at_2_in_24_of_24_pairs`, `test_O7_view_A_with_lead_time_1_costs_the_same_as_view_B_with_lead_time_2` |
| Was das Lager bestellt (Hauptzelle, Zufalls-Seed 11) | Die Echelon-Politik bestellt beim Lieferanten in praktisch jeder Periode die Nachfrage der letzten Periode (Quote 1,000), Sicht A die der vorletzten in rund 85 % der Perioden (Quote 0,848 über 200.000 Perioden; die App zeigt 20.000 Perioden: 0,85; je Zufalls-Seed 0,845 bis 0,853 bei 20.000 Perioden). | `test_claim_order_quotas_main_cell_seed_11` (Band; 200.000 Perioden: Messung dieser README) |
| Mehrbestand | Hauptzelle, exakt: mittlerer Lagerbestand Echelon 3,67, Sicht A 6,00, Sicht B 5,00; das Lager kann die Händlerbestellung in 35,3 % der Zeit (Echelon), 10,6 % (A), 11,1 % (B) nicht voll bedienen. Die Regel A hält in **48 von 48** Zellen mehr Lagerbestand als die Echelon-Politik; als Ersatz für die fehlende Sicht gedeutet (**Deutung**). | `test_claim_cost_breakdown_and_stocks_in_the_main_cell`, `test_claim_view_A_holds_more_warehouse_stock_than_echelon_in_48_of_48_cells` |
| Kostenanteile Hauptzelle | Echelon: Lager 1,4688 + Händlerbestand 3,5083 + Rückstand 1,6241 = 6,6012; A 2,4000 + 3,8942 + 1,0669; B 2,0000 + 3,9089 + 0,9641 (je Periode, exakt). | `test_claim_cost_breakdown_and_stocks_in_the_main_cell` |
| Niveaus | `S1` liegt zwischen 8 und 17, `S2 − S1` zwischen 0 und 9. `S1` hängt nicht an der Lagerlieferzeit (24 von 24 Paaren), wächst mit der Händlerlieferzeit (24 von 24), `S2 − S1` wächst mit der Lagerlieferzeit (24 von 24). In zwei Zellen ist `S2 = S1` (1/1 und 2/1, `h_w` 0,8, p 4): das Lager reicht nur durch. | `test_claim_levels_by_lead_times` |
| Rechenlast | Zustände der gemeinsamen Rechnung (Raster: Händlerbestand −20 bis S1 + 7, Lager 0 bis S2 − S1 + 16, 14 Werte je Pipelineplatz): 760 (1/1), 12.768 (1/2), 208.544 (1/3), 11.760 (2/1), 197.568 (2/2), 11.437.365.184 (4/4, nur gezählt). Faktor je zusätzlicher Periode Lieferzeit: 14,6- bis 16,9-fach über 24 Schritte. Die Zerlegung: 133.940 Rechenschritte (1/1) bis 1.218.854 (4/4). Zustände und Rechenschritte sind verschiedene Einheiten: die gemeinsame DP braucht je Iteration Zustände mal Aktionen Schritte. | `test_claim_state_counts_and_growth_factors` |
| Rechenzeit (Größenordnung) | Zerlegung: Millisekunden bis 4/4. Gemeinsame DP: unter 0,1 s (1/1), bis 0,3 s (1/2, 2/1), rund 5 bis 10 s (1/3, 2/2, 3/1); darüber nicht gerechnet (die Indextafel der gewählten Bauart ginge bei 4/4 in den Terabyte-Bereich). Zeiten sind unter Last gemessen und keine festen Zahlen. | **Messung der Vorarbeit, nicht im Test** |
| Bestellgrenze (1/1, `h_w` 0,4, p 9) | Beste Echelon-Politik (Aufzählung, neu abgestimmt) = DP für Grenze 4, 5, 6, 7, 8, 10, 13 (Zeitanteil an der Grenze 49 %, 26 %, 12 %, 5,6 %, 2,0 %, 0,18 %, 2,6e-5). Die Niveaus der Zerlegung (ohne Grenze gerechnet) verlieren bei Grenze 4 / 5 / 6 +7,71 / +1,28 / +0,05 %, ab 7 nichts. **Grenze 3 ist unbrauchbar** (Grenze gleich dem Nachfragemittel, Randmasse 4,5e-2). | Grenzen 5 und 8: `test_claim_order_limit_sweep_main_cell`; die übrigen: `scripts/qsweep.py` (**Messung dieser README, nicht im Test**) |
| Handbeispiel | Zerlegung `S1 = 3`, `S2 = 4`, `g = 31/16 = 1,9375`; Kette der Politik (3, 4) mit sechs wiederkehrenden Zuständen; lokal (beste Niveaus, Aufzählung): A +18,05 % (3, 3), B +16,13 % (3, 2), C 0,00 % (3, 1). | `test_claim_hand_example`, `test_O4_hand_example_exactly_with_fractions` |
| Simulation gegen exakt | Simulation (4 Zufalls-Seeds, je 61.000 Perioden) liegt für Echelon, A und B in der Hauptzelle innerhalb von vier Standardfehlern um den exakten Wert (Band, kein Gleichheitstest). In der Vorarbeit lagen alle neun Simulationsmittel (3 Lieferzeiten × 3 Regeln, je 4 Zufalls-Seeds × 1e6 Perioden) über dem exakten Wert, um 0,003 bis 0,012 (höchstens 1,91 Standardfehler; die Zeilen teilen Zufallsströme); eine Gegenprobe mit 16 neuen Zufalls-Seeds (Echelon 1/1) ergab 6,60039 gegen exakt 6,60129 (0,37 Standardfehler): kein Hinweis auf eine Verzerrung. | `test_O3_simulation_matches_exact_value_within_four_standard_errors`; Vorarbeit: **Messung der Vorarbeit, nicht im Test** |

## Ehrliche Grenzen

- **Nur dieses Modell.** Die Optimalität der Zerlegung ist als Satz an Voraussetzungen gebunden: seriell, Rückstand nur beim Endkunden, deterministische Lieferzeiten, stationäre unabhängige Nachfrage, keine Fixkosten, keine Kapazitätsgrenzen, zentrale Steuerung (van Houtum 2006, Chao/Zhou 2006, nach den frei lesbaren Fassungen). Die Demo misst sie im genannten Modell; **nicht gemessen**: Rückstand auch beim Lager, stochastische Lieferzeiten, mehr als zwei Stufen, Fixkosten, andere Nachfrageverteilungen, ein anderes Mittel als 3, ein anderes `h_r`.
- **Bestellgrenze.** Die Zerlegung ist ohne Grenze hergeleitet; mit Grenze 13 greift sie in höchstens 3,2e-5 der Zeit. Die Gleichheit mit der gemeinsamen DP wurde in der Hauptzelle auch für die Grenzen 4, 5, 6, 7, 8, 10 und 13 gemessen (mit neu abgestimmten Niveaus), für andere Zellen **offen**. Die Niveaus der Zerlegung selbst verlieren bei Grenze 4 / 5 / 6 +7,71 / +1,28 / +0,05 %.
- **Die Lücken der lokalen Regeln sind eigene Messungen.** Keine gelesene Quelle nennt eine Zahl für den Verlust einer lokalen Base-Stock-Regel. Eine Zahl wie die 40 % eines dezentralen Netzes in einer Arbeit zu gelernten Politiken gehört zu einem anderen Szenario (verlorene Nachfrage) und ist nicht vergleichbar.
- **Lokale Niveaus per Bergsteigen**, nicht als globales Optimum bewiesen (in Lieferzeit 1/1 gegen das volle Fenster geprüft, 24 von 24 gleich).
- **„Informationsverzug“ ist eine Deutung.** Gemessen ist die Gleichheit A bei Lagerlieferzeit 1 = B bei Lagerlieferzeit 2 (24 von 24 Paaren); dass der Grund eine Periode fehlender Information ist, folgt daraus nicht streng. Dasselbe gilt für „Mehrbestand als Ersatz“: gemessen ist nur, dass A mehr Lagerbestand hält.
- **Sicht C = Echelon-Politik** ist eigene Herleitung und Messbefund (gleiche Aktionen auf den erreichten Zuständen und gleiche Kosten in allen geprüften Fällen), **nicht** aus einer gelesenen Quelle zitiert.
- **Die unkoordinierte Wahl der Niveaus** (jede Stufe rechnet ihr Niveau für sich, ohne die induzierte Kostenfunktion der anderen) ist **nicht gemessen**: die Heuristik wäre beliebig. „Jede Stufe für sich“ meint hier die lokale Regel mit den besten Niveaus.
- **Randfall `h_w` = Händlersatz** (Echelon-Satz der Händlerstufe 0, `S1` nicht eindeutig) nicht gerechnet; der Regler endet bei 0,8.
- **Literatur.** Clark/Scarf (1960) ist nur nach dem Abstract bekannt (die Volltextseite war nicht zugänglich); die Struktur der Zerlegung stützt sich auf die Lektüre von van Houtum (2006, Arbeitsfassung der TU Eindhoven), Chao/Zhou (2006) und Zijm/Van Houtum (1994). Die Formeln für `L1` und `G2` sind eigene Herleitung, numerisch geprüft.
- **Rechenzeiten** sind Größenordnungen unter Last, keine festen Zahlen; die gemeinsame DP bei Lieferzeit 2/2 braucht rund 200 MB Arbeitsspeicher und auf einem Server länger als hier.

## Verifikation

- **Einheiten von Hand** (`tests/test_units.py`): Nachfrage über zwei Perioden, `L1`, `G2`, `S1`, `S2`, `g`, Kosten und Folgezustand, Bestellregeln, stationäre Verteilung der Kette (1/16, 3/16, 1/8, 3/8, 1/16, 3/16), Zustandszahl 13 · 7 = 91, alle am Handbeispiel gerechnet.
- **Orakel** (`tests/test_oracle_lager_und_haendler.py`): gemeinsame DP gegen Echelon-Politik (Kosten und Aktionen, fünf Zellen); Formel gegen exakte Kette und Aufzählung; Simulation mit eigener Regelschleife gegen die exakte Auswertung; Handbeispiel exakt mit Brüchen unabhängig vom Code; die neue DP gegen eingefrorene Werte der älteren Rechnung (dünne Matrix je Aktion); direkte Lösung gegen Potenziteration; Sicht C gegen Echelon; Sicht A bei Lw gegen Sicht B bei Lw + 1.
- **Eigenschaften und Zweige** (`tests/test_properties.py`): Warenerhaltung in jedem Verlaufsschritt, Konvexität der Lagerstufe, Unabhängigkeiten der Niveaus, jeder Regler und jede Sicht ändert das Ergebnis, Zeitanteile summieren sich zu 1.
- **Randfälle** (`tests/test_edge_cases.py`, `tests/test_algorithm.py`): konstante Nachfrage (von Hand 0,8), sehr kleines `p`, Bestellgrenze 1, gleiche Lagerkostensätze, absichtlich zu enges Raster (beide Ränder), kleinstes Raster.
- **Verdikte** (`tests/test_verdicts.py`): Schwellen und jede Zahl im Text an handgesetzten Werten. **Claims** (`tests/test_claims.py`, `tests/test_grid.py`): jede Zahl der Tabelle oben und die Datei `data/lah_grid.json` gegen eine Neuberechnung.
- **Oberfläche:** AppTest aller Presets, Lieferzeit-Kombinationen, Sichten, Regler an den Grenzen, Permalink, Handbeispiel, DP-Knopf bei 2/2 (`tests/test_app.py`), Layout und Footer, Laufzeit-Importe. 178 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfrage, vier Experimente, Handbeispiel, Expander |
| `lah_algorithm.py` | Kern: Zelle zusammensetzen, Raster wachsen lassen, beide Ränder prüfen |
| `lah_chain.py` | exakte Auswertung einer Politik auf der Markov-Kette, Randmassen |
| `lah_constants.py` | Regler-Grenzen, gemessene Zellen, feste Annahmen, Farben, Presets |
| `lah_decompose.py` | Zerlegung: `L1`, `G2`, `S1`, `S2`, geschlossene Kosten einer Echelon-Politik |
| `lah_demand.py` | Poisson-Verteilung, Faltung über mehrere Perioden |
| `lah_dp.py` | gemeinsame Rückwärtsrechnung (relative Wertiteration) |
| `lah_evaluation.py` | Einstellungen, Analyse, Verlauf, Verdikte als reine Funktionen |
| `lah_example.py` | Handbeispiel (Nachfrage 0, 1, 2) mit Code-Gegenprobe |
| `lah_layout.py` | Fester Kopf (Stylesheet) |
| `lah_model.py` | Parameter, Kosten der Periode, Folgezustand, Zustandsraster |
| `lah_pdf_export.py` | PDF-Bericht der gewählten Zelle |
| `lah_policies.py` | Echelon-Base-Stock und die drei Sichten der lokalen Regel |
| `lah_presets.py` | Permalink (Begrenzen, Einrasten), Presets, Zufalls-Seed-Button |
| `lah_results.py` | die 48 vorgerechneten Zellen und die Zählungen darüber |
| `lah_scaling.py` | Rechenlast-Tafel (Zustandszahl gegen Aufwand der Zerlegung) |
| `lah_simulate.py` | SplitMix64, Nachfragestrom, Verlauf einer Regel |
| `lah_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `data/lah_grid.json` | die 48 Zellen, vorgerechnet |
| `tools/build_grid.py` | erzeugt `data/lah_grid.json` (einige Minuten) |
| `scripts/qsweep.py` | Messung zur Bestellgrenze (README, nicht im Test) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Einheiten von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Claims, Grid, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Regler für Lieferzeit 3 und 4:** die gemeinsame Rechnung ist dort nur noch auf Raster bis Lieferzeit-Summe 4 gerechnet; die Rechenlast-Tafel zeigt sie als Zahl, nicht als Zelle.
- **Freie Niveaus der lokalen Regel:** die Niveaus der Sichten A, B und C stehen fest (vorgerechnet); frei wählbar sind nur die Abweichungen der Echelon-Niveaus (geschlossene Formel, ohne Bestellgrenze).
- **Handbeispiel als Preset:** das Handbeispiel hat eine andere Nachfrageverteilung; als Preset würden `h_w` und `p` zu wirkungslosen Reglern. Es steht deshalb als eigener Abschnitt mit festen Werten.
- **Drei und mehr Stufen, Rückstand im Lager, stochastische Lieferzeiten, Fixkosten:** nicht gerechnet (siehe Grenzen).
- **Erklärseite „verständlich erklärt“ und Website-Einbindung:** noch nicht angelegt.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Bestandspolitiken: von Base-Stock bis Regelwahl](https://sebastianhanisch.net/konzepte-bestandspolitiken.html).
