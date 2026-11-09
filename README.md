# Auftragsallianz: wie viel sparen Spediteure, wenn sie Aufträge zusammenlegen? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-auftragsallianz-demo.streamlit.app/)**

Stück der Linie **Gemeinsam gewinnen** in der „Konzepte“-Reihe für die Website
„Sebastian Hanisch – Operations Research und Machine Learning“. Es führt die Linie zusammen: Wo die Vorgänger Bestände zusammenlegen, geht es hier um **Wege**: Spediteure mit eigenen
Depots legen ihre Stopps zusammen. Die Auswahl der Aufträge ist zugleich ein Tausch von Punkten mit verschiedenem Wert. Die Aufteilungstheorie steht in
[kern-demo](https://sebastianhanisch-kern-demo.streamlit.app/) und [shapley-demo](https://sebastianhanisch-shapley-demo.streamlit.app/) und wird hier nicht neu gebaut; der Kern dient nur als Kennzahl.

## Kernfrage

*Wie viel sparen Spediteure mit eigenen Depots, wenn sie ihre Stopps zusammenlegen, wovon hängt die Ersparnis ab, und welche Aufträge gibt man in den Pool?*

## Modell

**Vehikel** (`aal_scenario.py`): $n$ Spediteure (2 bis 5), jeder mit einem eigenen Depot im Einheitsquadrat und **einem Fahrzeug ohne Kapazität**, je $k$ Stopps (1 bis 6; höchstens 12 Stopps insgesamt).
Die Stopps liegen **normalverteilt um das eigene Depot** mit Standardabweichung $s$ (Regler „Streuung“, 2 bis 99 % der Gebietsbreite, nicht auf das Gebiet beschnitten) oder bei 100 %
**gleichverteilt im Einheitsquadrat**. Auf Wunsch haben einige Spediteure keine eigenen Stopps („ohne eigene Aufträge“): sie stellen nur Depot und Fahrzeug. Entfernungen sind euklidisch.
Der Zufall kommt aus einer wörtlichen Kopie von SplitMix64 (`aal_rng.py`, Original `ab-test-demo/abt_rng.py`, Kopietest `tests/test_copies.py`); je Stopp werden immer Normal- und Gleichverteilte gezogen,
sodass Instanzen mit gleichem Zufalls-Seed und verschiedener Streuung gekoppelt sind (gleiche Depots, gleiche Zufallszahlen).

**Tourkosten** (`aal_algorithm.py`): kürzeste Rundtour ab dem Depot über jede Teilmenge der Stopps, exakt per **Held-Karp** (Bitmasken). **Koalitionskosten** $C(S)$: die Stopps aller Mitglieder werden bestmöglich
auf die Fahrzeuge der Mitglieder verteilt, exakt per Teilmengen-DP über die Koalitionen. **Ersparnis** = $1 - C(N) / \sum_i C(\{i\})$ gegen „jeder fährt allein“. Das ist die Ersparnis bei voller Information
und zentraler Planung, in diesem Modell also eine Obergrenze für jedes Verfahren, das Stopps auf die Fahrzeuge verteilt.

**Aufteilung** (`aal_game.py`): Shapley-Kostenanteile (exakt aus den Koalitionswerten), proportional zu den Alleinkosten, und „im Kern?“ per linearem Programm (HiGHS, scipy). **Pool** (`aal_pool.py`): jeder Spediteur
mit Aufträgen gibt genau einen Stopp ab; die abgegebenen Stopps werden danach bestmöglich auf alle Fahrzeuge verteilt (alle Zuordnungen), die übrigen bleiben bei ihrem Spediteur. Regeln für die Wahl: **teuerster**
(größte Entlastung der eigenen Tour), vier geografische Lesarten (**nächster an einem fremden Depot**, **nächster an einem fremden Stopp**, **am weitesten vom eigenen Depot**, **am weitesten von den eigenen anderen Stopps**),
**zufällig** (der Erwartungswert über alle Wahlen, jede gleich wahrscheinlich) und **bestmöglich** (alle Wahlen). Die Formeln stehen im 📐-Expander der App.

## Methodik

- Alles ist exakt gerechnet, nichts geschätzt. Die Experimente laufen erst auf Klick über feste Instanzen (Zufalls-Seeds `100000` aufwärts, in Experiment 3 zusätzlich `200000` und `400000` aufwärts) mit Spinner und Zeitangabe.
- **Orakel** (`tests/test_oracle_auftragsallianz.py`): (1) Held-Karp gegen Aufzählung aller Permutationen, (2) Koalitionskosten gegen Vollaufzählung aller Zuordnungen (Touren per Permutationen), (3) Kern-LP gegen das duale
  Programm des Satzes von Bondareva und Shapley und gegen eine Gittersuche bei drei Spediteuren, (4) Shapley gegen das Mittel über alle Permutationen, (5) bestmögliche Pool-Wahl gegen Vollaufzählung von Wahl und Zuordnung.
  Die Orakel selbst sind an Handbeispielen geprüft (Tausch-Instanz mit $C = (0, 18, 18, 4)$, 3-4-5-Dreieck, ein Spiel mit leerem Kern und Least-Core-Wert 2). Im Scratch-Lauf (nicht im Repo, nicht im Test) lagen je Orakel
  200 Zufallsinstanzen ohne eine Abweichung (Held-Karp: 200 Instanzen mit allen Teilmengen; Koalitionen: 200 Instanzen mit allen 7 Koalitionen; Kern: 400 Instanzen mit 3 und 4 Spediteuren gegen das Dual, 50 davon gegen das Gitter;
  Shapley: 200; Pool: 200).
- **Handrechnungen** (`tests/test_algorithm.py`): 3-4-5-Dreieck (6, 10, 12), Tausch-Instanz ($C = (0, 18, 18, 4)$, Shapley und proportional je $(2, 2)$, Least-Core-Wert $-16$), Spiel mit leerem Kern, Auswahlregeln auf einer Linie.
- **Randfälle** (`tests/test_edge_cases.py`): zwei Spediteure, ein Stopp, Spediteure ohne Aufträge (ihr Shapley-Kostenanteil ist höchstens 0), Stopps auf demselben Punkt oder auf dem Depot, Reglerränder, der größte Fall
  (4 Spediteure mit je 3 Stopps), Reproduzierbarkeit über Prozesse mit verschiedenem Hash-Seed.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind Mittel oder Zählungen über feste Instanzen mit exakter Rechnung im Vehikel dieser Demo (Fahrzeuge ohne Kapazität, Zufallsinstanzen im Einheitsquadrat, volle Information). „Ersparnis“ meint Ersparnis gegen „jeder fährt allein“.

| Frage | Befund | Test |
|---|---|---|
| Ersparnis über die Streuung, 4 Spediteure mit je 2 Stopps, 200 Instanzen | Streuung 5 %: Mittel 3,7 %, Median 0,0 %, in **138 von 200** Instanzen keine Ersparnis. 15 %: Mittel 19,7 %, Median 18,6 %, 16 von 200 ohne. 30 %: 33,5 %, Median 33,8 %, 1 von 200 ohne. Gleichverteilt: 59,5 %, Median 60,1 %, 0 von 200 ohne. Das Mittel steigt in allen drei Übergängen zwischen den vier Stufen. | `test_saving_over_the_overlap_4_carriers_2_stops_200_instances` |
| Dasselbe bei 3 Spediteuren mit je 3 Stopps (100 Instanzen) | Mittel 2,1 % (84 ohne Ersparnis), 14,0 % (21 ohne), 26,8 % (2 ohne), 52,8 % (0 ohne) bei Streuung 5 %, 15 %, 30 %, gleichverteilt. | `test_saving_over_the_overlap_3_carriers_3_stops_100_instances` |
| Dasselbe bei 5 Spediteuren mit je 2 Stopps (60 Instanzen) | Mittel 4,5 % (34 ohne Ersparnis), 21,8 %, 37,9 %, 66,7 % (je 0 ohne). | `test_saving_over_the_overlap_5_carriers_2_stops_60_instances` |
| Ist der Kern leer? | In **keiner** der 1 440 Instanzen dieser drei Messreihen (4 Stufen mal 360 Instanzen) ist der Kern leer. Das ist kein Beweis, dass er in diesem Vehikel nie leer ist, und sagt nichts über Touren mit Kapazitäten oder Zeitfenstern. | `test_the_core_is_never_empty_in_1440_instances` |
| Liegen Shapley und proportionale Aufteilung im Kern? (4 Spediteure, 2 Stopps, 200 Instanzen je Stufe) | Shapley im Kern: 200, 164, 151, 180 von 200 (Streuung 5 %, 15 %, 30 %, gleichverteilt), also **nicht immer**. Proportional: 139, 22, 48, 137 von 200. Bei Streuung 5 % sind 138 dieser Instanzen ohne Ersparnis; dort liegt die Aufteilung trivial im Kern, die Zahlen 200 und 139 sind davon geprägt. | `test_shapley_and_proportional_in_the_core_4_carriers_2_stops` (Band ± 2) |
| Was bringt jedes weitere Mitglied? (je 2 Stopps, Streuung 15 %, 100 Instanzen) | Mittel bei 2, 3, 4, 5 Spediteuren: 9,5 %, 13,9 %, 18,2 %, 22,1 % (Median 0,0 %, 12,5 %, 15,9 %, 21,9 %). Zuwachs gegenüber dem Vorgänger 4,5, 4,3, 3,9 Prozentpunkte: in diesen drei Schritten sinkend, aber nur um 0,2 bis 0,4 Prozentpunkte; bei 100 Instanzen ist das nicht abgesichert, die Streuung der Mittel kann die Reihenfolge kippen. Mit jedem Mitglied kommen auch Depot, Fahrzeug und Stopps dazu. | `test_marginal_gain_per_additional_carrier_100_instances` |
| Welche Aufträge in den Pool? 4 Spediteure, je 3 Stopps, Streuung 15 %, 60 Instanzen | Mittel der Ersparnis: zufällig 6,7 %, **teuerster 13,1 %**, nächster an fremdem Depot 10,4 %, nächster an fremdem Stopp 9,4 %, weit vom eigenen Depot 12,4 %, von den eigenen anderen Stopps entfernt 12,7 %, bestmöglich 14,8 %. | `test_pool_selection_4_carriers_3_stops_spread_15_60_instances` |
| Dasselbe bei Streuung 30 %, 60 Instanzen | zufällig 11,0 %, **teuerster 19,7 %**, fremdes Depot 13,5 %, fremder Stopp 13,4 %, weit vom Depot 18,1 %, isoliert 19,6 %, bestmöglich 23,2 %. Gegen den teuersten Auftrag (besser, gleich, schlechter): fremdes Depot 8, 0, 52; fremder Stopp 9, 3, 48; weit vom Depot 11, 23, 26; isoliert 11, 39, 10. | `test_pool_selection_4_carriers_3_stops_spread_30_60_instances` |
| Stabilität der Reihenfolge (Streuung 30 %, anderer Seed-Bereich, 200 Instanzen) | Mittel: zufällig 11,0 %, teuerster 19,7 %, fremdes Depot 14,2 %, fremder Stopp 14,0 %, weit vom Depot 18,3 %, isoliert 19,2 %, bestmöglich 22,9 %; die Reihenfolge der Mittel ist wie bei 60 Instanzen. Der Abstand zwischen „teuerster“ und „isoliert“ ist klein: 0,5 Prozentpunkte bei einem Standardfehler von 0,3 (200 Instanzen), 0,1 bei 0,4 (60 Instanzen); als Tendenz zu lesen, nicht als gesicherter Unterschied. | `test_pool_selection_spread_30_with_200_other_instances_gives_the_same_order` |
| Zwei Stopps je Spediteur | „Am weitesten vom eigenen Depot“ und „teuerster“ erzielen in allen 200 Instanzen (Streuung 30 %, 4 Spediteure) dieselbe Ersparnis (gemessen wird die Ersparnis, nicht der gewählte Stopp; für andere Streuungen nicht gemessen). Die Lesart „von den eigenen anderen Stopps entfernt“ hat dann zwei gleich weit entfernte Stopps und wählt den mit kleinerem Index. | `test_with_two_stops_per_carrier_the_distance_to_the_depot_rule_equals_the_most_expensive_rule` |
| Nur ein Spediteur hat Aufträge (4 Stopps, 3 Spediteure ohne Aufträge, 100 Instanzen) | Streuung 30 %: Mittel 10,6 %, 34 von 100 ohne Ersparnis. Gleichverteilt: Mittel 24,4 %, 8 von 100 ohne. Ihr Shapley-Kostenanteil ist in 15 getesteten Instanzen nie positiv (ein Zusatzfahrzeug kann ungenutzt bleiben, die Koalitionskosten steigen durch es nicht). | `test_only_one_carrier_has_orders_4_stops_100_instances`, `test_an_idle_carrier_never_pays_more_than_nothing` |
| Presets (je eine Instanz mit dem Preset-Seed) | Voreinstellung 14,1 %, Eng um die Depots **0,0 %**, Gleichverteilt 46,3 %, Fünf Spediteure 12,6 % (dort liegt Shapley nicht im Kern), Nur ein Spediteur 13,5 %. Eine Instanz ist ein Beispiel, kein Mittel. | `test_every_preset_runs_through_the_core_question_with_the_expected_verdict` |

## Befunde und Korrekturen gegenüber dem Plan

- **Ersparnis über die Überlappung bestätigt:** Der Plan nannte für 4 mal 2 Stopps 2,6, 17,9, 32,0 und 59,8 % (200 Instanzen, andere Zufallszahlen, Streuungen 0,05, 0,15, 0,3 und gleichverteilt). Mit dem Zufallsgenerator
  der Demo (SplitMix64) und denselben Stufen kommen 3,7, 19,7, 33,5 und 59,5 %: dieselbe Reihenfolge und Größenordnung, die Einzelzahlen weichen ab (gekoppelte Instanzen, anderer Zufall). Nur die Zahlen dieser README gelten.
- **Kern in 1 440 Instanzen nie leer (Plan: in 410):** 0 von 1 440; kein Beweis.
- **Neu gegenüber dem Plan:** Der Shapley-Wert liegt nicht immer im Kern (36 von 200 Instanzen bei Streuung 15 % außerhalb), die proportionale Aufteilung liegt bei Streuung 15 % und 30 % in 22 und 48 von 200 Instanzen im Kern, bei Streuung 5 % und gleichverteilt in 139 und 137 von 200.
- **Pool-Auswahl:** Die Zahlen der Vormessung im Plan sind mit dem heutigen Code nicht reproduzierbar und stehen deshalb nicht hier. Mit den festen 30 Instanzen (Zufalls-Seeds `100000` bis `100029`, 4 Spediteure mit je 3 Stopps, Streuung 30 %) kommen: zufällig 11,4 %, nächster an fremdem Depot 13,5 %, teuerster 21,2 %, bestmöglich 23,9 % (mit 60 Instanzen: 11,0, 13,5, 19,7, 23,2 %); die Reihenfolge ist dieselbe wie im Plan. „Zufällig“ ist hier der Erwartungswert über alle Wahlen, nicht eine einzelne Ziehung.
- **V4b, Aussage der Literatur nicht bestätigt (in diesem Modell, mit diesen Lesarten):** Gansterer und Hartl schreiben im Newsletter (IFORS News, Dezember 2016) und im Überblick (arXiv 1706.05254), die besten Kriterien zur
  Bewertung von Aufträgen für den Pool nähmen geografische Aspekte in den Blick und schlügen reine Gewinnstrategien deutlich. **Nur Newsletter und Überblick sind gelesen**, nicht der Aufsatz, auf den sie verweisen. Der Überblick zeigt in einer Abbildung einen Spediteur, der nach dem Grenzgewinn wählt, und zwei, die geografische
  Informationen einbeziehen; die Kriterien selbst werden in keinem der beiden Texte definiert. Nachgebaut sind darum vier **eigene Lesarten** von „geografisch“ und eine von „Gewinn“ (die Entlastung der eigenen Tour).
  **Zählung der gemessenen Einstellungen:** Bei 3 Stopps je Spediteur liegt „teuerster“ im Mittel **vor jeder** geografischen Lesart (Reihenfolge der Mittel, nicht gesicherter Unterschied): Streuung 15 % (60 Instanzen), 30 % (60 und 200 Instanzen); in Scratch-Läufen
  (nicht im Test) ebenso bei Streuung 100 % (100 Instanzen) und bei 3 Spediteuren mit je 3 Stopps (200 Instanzen, Streuung 15 % und 30 %). Gegen fremdes Depot und fremden Stopp ist der Vorsprung deutlich (mehrere Prozentpunkte); gegen „von den eigenen anderen Stopps entfernt“ (isoliert) ist er klein und liegt innerhalb der Streuung: Abstand 0,1 bis 0,5 Prozentpunkte bei Standardfehlern von 0,3 (4 mal 3 Stopps, Streuung 15 % und 30 %, 60 und 200 Instanzen). Bei 2 Stopps je Spediteur (4 Spediteure, 200 Instanzen, und 5 Spediteure, 100 Instanzen, Scratch-Läufe) liegt
  „teuerster“ vor drei Lesarten und **gleichauf** mit „weit vom eigenen Depot“, das dort dieselbe Ersparnis erzielt (Test dazu in der Tabelle oben). Einzelne Instanzen gehen zugunsten der geografischen Lesart aus (8 bis 11 von 60 bei
  Streuung 30 %). **Das Ergebnis hängt an der Operationalisierung:** Die Literatur arbeitet mit Kapazitäten, Auktionen und Gewinnen je Auftrag; hier gibt es keines davon, und „teuerster“ ist die direkt auf die Kosten
  gerichtete Regel. In diesem Modell ist keine der vier Lesarten klar besser als der teuerste Auftrag; gegen „isoliert“ liegt der Abstand innerhalb der Streuung. Das widerlegt die Aussage der Literatur nicht. Das Team-Orienteering-Kriterium (Aufträge, die nicht in der besten Tour liegen, kommen in den Pool) ist nicht nachgebaut, weil es Gewinne je Auftrag und
  begrenzte Fahrzeuge braucht.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| Ein Fahrzeug ohne Kapazität je Spediteur | Mit Kapazitäten, Zeitfenstern oder Abhol-und-Liefer-Paaren ändern sich Tourkosten und Koalitionswerte; hier nicht gemessen. | vrp-demo (Tourenplanung mit Kapazitäten) |
| Alles exakt, bis 12 Stopps | Der Aufwand wächst mit $3^M$ je Koalition; darüber braucht es Heuristiken, die hier fehlen. | – |
| Eine Zentrale kennt alle Stopps und Kosten | Ohne volle Information wird die beste Verteilung nicht erreicht; die Ersparnis hier ist eine Obergrenze. | auction-demo, task-swap-demo |
| Jeder gibt genau einen Stopp ab | Die Zahl der abgegebenen Stopps ist selbst eine Entscheidung; hier nicht untersucht. | – |
| Stopps streuen normalverteilt um das Depot | Echte Kundenlagen sind geklumpt; die Streuung ist eine Stellvertreter-Größe für die Überlappung. | – |
| Keine Zahlungen, keine Anreize | Ob Spediteure ehrlich auswählen, ist nicht modelliert. | – |

Weitere Grenzen: Die Mittel der Experimente hängen an den festen Seed-Bereichen (siehe die Zahl der Instanzen neben jeder Zahl); die Anzeige in der App rechnet je nach Regler 10 bis 60 Instanzen. Die geografischen Auswahlregeln sind eigene Lesarten. Die Anzeige ist im
Browser an Presets und am Regler-Maximum geprüft, nicht auf jedem Gerät. Laufzeit: Ein Experiment am größten Fall (4 Spediteure mit je 3 Stopps, 60 Instanzen, 4 Stufen) rechnet auf dem Entwicklungsrechner 14 bis 17 s (zwei Messungen, eine davon im Browser; Messung dieser README, nicht im Test).

## Verifikation

- **Kern:** Handrechnungen, fünf Orakel, Randfälle (siehe Methodik); unabhängiger Zulässigkeitsprüfer der Zuordnung läuft auch in der App.
- **Verdikte:** reine Funktionen in `aal_evaluation.py`, Schwellen und Zahlen im Text in `tests/test_verdicts.py`; jedes Preset durchläuft die Kernfrage (`tests/test_presets.py`).
- **Oberfläche:** AppTest (jeder Button, jeder Regler an Minimum und Maximum, alle Minima und alle Maxima zusammen, Permalink mit Begrenzung, Experimentknöpfe, Presets), Layout (Schnellstart, fester Kopf, Hervorhebung), Footer, Laufzeit-Importe gegen `requirements.txt` (scipy ist Laufzeitbibliothek).

## Tests

105 Tests (`pytest --collect-only -q tests`), Laufzeit lokal rund 24 s mit `pytest tests -q -n 8` und rund 60 s ohne `-n` (eine Messung je, Python 3.14, Windows). Darunter Orakel (8), Claims (11, rund 32 s allein), Verdikte (12), Presets (13), Randfälle (18), AppTest und Layout (24).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `aal_algorithm.py` | Kern 1: Held-Karp, Verteilung auf Fahrzeuge, Koalitionskosten, Rückverfolgung |
| `aal_game.py` | Kern 2: Shapley, proportional, Kern per LP |
| `aal_pool.py` | Kern 3: Pool-Auswahl, Auswahlregeln, Ersparnis je Regel |
| `aal_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `aal_evaluation.py` | Analyse, Ensembles über feste Instanzen, Verdikte |
| `aal_layout.py` | Fester Kopf (Stylesheet) |
| `aal_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `aal_rng.py` | Wörtliche Kopie von SplitMix64 aus `ab-test-demo` |
| `aal_scenario.py` | Instanzen: Depots und Stopps |
| `aal_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `app.py` | Streamlit-App (nach DEMO-PLAYBOOK §2) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt), darunter scipy |
| `tests/` | Mini-Instanzen und Handrechnungen, Orakel, Claims, Verdikte, Randfälle, Presets, Layout, AppTest, Footer, Kopie, Laufzeit-Importe |

## Bewusst nicht umgesetzt

Kapazitäten, Zeitfenster, Abhol-und-Liefer-Paare, mehrere Fahrzeuge je Spediteur, Anreizverträglichkeit der Auktionen, Zahlungen zwischen Spediteuren, eine Heuristik für mehr als 12 Stopps (nur als Hinweis), ein
PDF-Export, das Team-Orienteering-Kriterium der Auftragsauswahl, eine eigene Aufteilungstheorie (Nukleolus und weitere: kern-demo).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: die dort eingestellte Python-Version ist nicht geprüft; CI läuft mit Python 3.12, lokal 3.14. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
