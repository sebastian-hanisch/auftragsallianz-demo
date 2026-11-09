"""Auftragsallianz: wie viel sparen Spediteure, wenn sie Aufträge zusammenlegen? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück der Linie „Gemeinsam gewinnen“. Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import aal_constants as C
import aal_evaluation as E
import aal_pool as P
from aal_evaluation import Settings, de
from aal_layout import dunkles_design, kopf_css
from aal_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from aal_visualization import build_map, build_members, build_overlap, build_pool, build_shares

st.set_page_config(page_title="Auftragsallianz – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return E.analyse(settings)


@st.cache_data(show_spinner=False)
def _pool_instance(settings):
    a = E.analyse(settings)
    return P.criterion_picks(a.solved), P.criterion_savings(a.solved)


def _seeds(count):
    return C.SWEEP_SEEDS[:count] if count <= len(C.SWEEP_SEEDS) else tuple(range(100000, 100000 + count))


def _clamp(key, lo, hi):
    """Bringt den Zustand eines Reglers vor dem Zeichnen in seine (von anderen Reglern abhängigen) Grenzen."""
    st.session_state[key] = max(lo, min(hi, int(st.session_state.get(key, lo))))


def _run_experiment(name, key, label, compute, spinner_text):
    """Experiment erst auf Klick rechnen (DEMO-PLAYBOOK §1c); Ergebnis, Einstellungsschlüssel und Rechenzeit stehen im Session State. Gibt (Ergebnis, Sekunden) oder None zurück."""
    if st.button(label, key=f"run_{name}"):
        with st.spinner(spinner_text):
            start = time.perf_counter()
            result = compute()
            st.session_state[f"exp_{name}"] = (key, result, time.perf_counter() - start)
    stored = st.session_state.get(f"exp_{name}")
    if stored is None:
        return None
    if stored[0] != key:
        st.caption("Die Einstellungen haben sich seit der letzten Rechnung geändert: bitte erneut rechnen.")
        return None
    return stored[1], stored[2]


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in aal_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Auftragsallianz: wie viel sparen Spediteure, wenn sie Aufträge zusammenlegen?")
st.markdown(
    f"""
**Spediteure mit eigenem Depot** legen ihre **Stopps zusammen**: In der Allianz darf jedes Fahrzeug jeden Stopp der Mitglieder fahren. Die Demo rechnet exakt (bis {C.MAX_STOPS} Stopps),
**wie viel das spart**, **wovon die Ersparnis abhängt** (Überlappung der Reviere, Zahl der Mitglieder), **wie man sie aufteilt** (Shapley, proportional, Kern als Kennzahl) und
**welche Aufträge man in den Pool gibt**. Stück der Linie „Gemeinsam gewinnen“: Die Wege treten an die Stelle der Bestände, die Auswahl der Aufträge ist ein Tausch
von Punkten mit verschiedenem Wert.

Weiter unten: die **Kernfrage**, vier Abschnitte **🔬 Experiment**, **🚧 Wo die Annahmen enden**, **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Spediteure und Aufträge**")
    n = st.slider("Spediteure n", *bounds("n_slider"), key="n_slider", help="Wie viele Spediteure die Allianz bilden; jeder hat ein Depot und ein Fahrzeug.")
    _clamp("idle_slider", C.IDLE_MIN, int(n) - 1)
    idle = st.slider("davon ohne eigene Aufträge", C.IDLE_MIN, int(n) - 1, key="idle_slider", help="Spediteure, die nur Depot und Fahrzeug haben und keine eigenen Stopps.")
    _clamp("per_slider", C.PER_MIN, C.max_per(int(n), int(idle)))
    per = st.slider("Stopps je Spediteur mit Aufträgen", C.PER_MIN, C.max_per(int(n), int(idle)), key="per_slider",
                    help=f"Die Gesamtzahl der Stopps ist auf {C.MAX_STOPS} begrenzt, damit alle Rechnungen exakt bleiben.")
    spread = st.slider("Streuung der Stopps um das eigene Depot (% der Gebietsbreite)", *bounds("spread_slider"), key="spread_slider",
                       help="Klein: die Stopps liegen nah am eigenen Depot. Groß: die Reviere überlappen. 100 heißt gleichverteilt im Gebiet.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.markdown("**Experimente**")
    ninst = st.slider("Zahl der Instanzen je Experiment", *bounds("ninst_slider"), key="ninst_slider", step=C.NINST_STEP,
                      help="Die Experimente rechnen erst auf Klick über so viele feste Instanzen; mehr Instanzen brauchen länger.")

sync_query_params({"n_slider": int(n), "per_slider": int(per), "spread_slider": int(spread), "idle_slider": int(idle), "ninst_slider": int(ninst), "seed_input": int(seed)})

settings = Settings(int(n), int(per), int(spread), int(idle), int(ninst), int(seed))
a = _analysis(settings)
inst, solved = a.solved.inst, a.solved
alone, grand = float(solved.alone.sum()), solved.grand
spread_text = "gleichverteilt im Gebiet" if settings.spread >= C.SPREAD_MAX else f"Streuung {settings.spread} %"
shapley_text = ", ".join(f"{de(x, 3)}" for x in a.shapley)
proportional_text = ", ".join(f"{de(x, 3)}" for x in a.proportional)
core_text = "leer" if a.core_empty else "nicht leer"

with st.expander("So funktioniert die Auftragsallianz", expanded=True):
    st.markdown(
        f"""
1. **Jeder allein:** Ein Spediteur fährt von seinem Depot die kürzeste Rundtour über seine Stopps (**Held-Karp**, exakt über alle Teilmengen).
2. **Die Allianz:** Eine Koalition legt die Stopps aller Mitglieder zusammen und verteilt sie bestmöglich auf die Fahrzeuge der Mitglieder (exakte Rechnung über alle Zuordnungen).
   Das gibt für jede Gruppe ihre Kosten $C(S)$.
3. **Ersparnis** = 1 − gemeinsame Kosten / Summe der Alleinkosten. Wie die Ersparnis von der Überlappung der Reviere abhängt, misst Experiment 1.
4. **Aufteilung:** Wer bekommt wie viel von der Ersparnis? Zwei Regeln und eine Kennzahl (Tabelle unten).
5. **Pool:** Gibt jeder Spediteur nur einen Stopp ab, entscheidet die Auswahlregel, welcher es ist.

Beispielspur: die aktuelle Instanz ({spread_text}, Zufalls-Seed {settings.seed}).

| Verfahren oder Regel | Idee | Beispielspur |
|---|---|---|
| **Held-Karp** | Kürzeste Rundtour über eine Stopp-Menge, Aufbau Teilmenge für Teilmenge | Spediteur 1 allein: Tour über {len(inst.stops_of(0))} Stopps kostet {de(solved.alone[0], 3)} |
| **Verteilung auf Fahrzeuge** | Beste Zuordnung der Stopps einer Koalition zu ihren Fahrzeugen | alle {inst.n} Spediteure gemeinsam: {de(grand, 3)} statt {de(alone, 3)} |
| **Shapley** | Jeder zahlt den mittleren Zusatzbeitrag, den er über alle Eintrittsreihenfolgen zu den Kosten der Gruppe leistet | Kostenanteile: {shapley_text} |
| **proportional** | Kostenanteil im Verhältnis der Alleinkosten | Kostenanteile: {proportional_text} |
| **Kern** | Aufteilungen, bei denen keine Gruppe mehr zahlt, als sie allein bräuchte; hier nur als Kennzahl (Theorie in kern-demo) | Least-Core-Wert {de(a.epsilon, 3)}; der Kern ist {core_text} |
| **Auswahlregeln** | Welchen Stopp gibt ein Spediteur in den Pool? Teuerster, nächster an fremdem Depot, und weitere (Experiment 3) | siehe Tabelle in Experiment 3 |
"""
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Gemeinsame Kosten", de(grand, 3), delta=f"{de(grand - alone, 3, True)} gegen allein", delta_color="inverse")
m2.metric("Alle fahren allein", de(alone, 3), delta="Referenz", delta_color="off")
m3.metric("Ersparnis", f"{de(E.saving_percent(grand, alone), 1)} %", delta="in % der Alleinkosten", delta_color="off")
st.plotly_chart(build_map(inst, E.routes(a)), width="stretch", key="main_map")
st.caption(
    f"Quadrate: Depots, Kreise: Stopps, jeweils in der Farbe des Spediteurs. Linien: die Touren der besten gemeinsamen Verteilung, in der Farbe des Fahrzeugs. "
    f"Ein Stopp in fremder Farbe an einer Tour heißt: Ein anderer Spediteur fährt ihn. Einheit: Kosten sind Strecken im Einheitsquadrat."
)

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viel sparen Spediteure mit eigenen Depots, wenn sie ihre Stopps zusammenlegen, und welche Aufträge gibt man in den Pool?")
kind, text = E.verdict_saving(alone, grand)
getattr(st, kind)(text)
st.caption("Das gilt für diese eine Instanz. Wovon die Ersparnis im Mittel abhängt und welche Aufträge in den Pool gehören, zeigen die Experimente über feste Instanzen.")

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie hängt die Ersparnis von der Überlappung der Reviere ab?")
st.caption(
    f"Vier Streuungsstufen ({', '.join(('gleichverteilt' if s >= C.SPREAD_MAX else f'{s} %') for s in C.SPREAD_STAGES)}), je {settings.ninst} feste Instanzen mit den Einstellungen "
    f"{settings.n} Spediteure, {a.settings.per} Stopps je Spediteur mit Aufträgen, {a.settings.idle} ohne Aufträge. Gleiche Zufalls-Seeds in allen Stufen (gleiche Depots, gleiche Zufallszahlen)."
)
key1 = (settings.n, a.settings.per, a.settings.idle, settings.ninst)
res1 = _run_experiment(
    "overlap", key1, "🧪 Experiment 1 rechnen",
    lambda: E.sweep_spread(settings.n, a.settings.per, a.settings.idle, _seeds(settings.ninst)),
    f"Rechne {len(C.SPREAD_STAGES) * settings.ninst} Instanzen exakt …",
)
if res1:
    rows1, secs = res1
    st.plotly_chart(build_overlap(rows1), width="stretch", key="chart_overlap")
    st.dataframe(
        [{"Streuung": ("gleichverteilt" if r["spread"] >= C.SPREAD_MAX else f"{r['spread']} %"), "Instanzen": r["count"], "Mittel": r["mean"], "Median": r["median"],
          "Minimum": r["min"], "Maximum": r["max"], "ohne Ersparnis": r["zero"]} for r in rows1],
        hide_index=True, width="stretch",
        column_config={k: st.column_config.NumberColumn(format="%.1f %%") for k in ("Mittel", "Median", "Minimum", "Maximum")},
    )
    kind, text = E.verdict_overlap(rows1)
    getattr(st, kind)(text)
    st.caption(f"Rechenzeit {de(secs, 1)} s.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wer gewinnt wie viel?")
st.caption(
    "Die Ersparnis der Allianz wird als Kostenanteile verteilt: nach **Shapley** und **proportional** zu den Alleinkosten. Der **Kern** ist nur eine Kennzahl: "
    "Liegt die Aufteilung im Kern, würde keine Gruppe allein besser fahren. Die Theorie der Aufteilung steht in kern-demo und shapley-demo."
)
st.plotly_chart(build_shares(solved.alone, a.shapley, a.proportional), width="stretch", key="chart_shares")
st.dataframe(
    [{"Spediteur": i + 1, "Stopps": len(inst.stops_of(i)), "Alleinkosten": float(solved.alone[i]), "Kostenanteil Shapley": float(a.shapley[i]),
      "Kostenanteil proportional": float(a.proportional[i])} for i in range(inst.n)],
    hide_index=True, width="stretch",
    column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ("Alleinkosten", "Kostenanteil Shapley", "Kostenanteil proportional")},
)
kind, text = E.verdict_shares(a)
getattr(st, kind)(text)
if res1:
    kind, text = E.verdict_shares_ensemble(res1[0])
    getattr(st, kind)(text)
else:
    st.caption("Die Zählung über viele Instanzen (Kern leer? Aufteilung im Kern?) kommt aus Experiment 1: dort rechnen.")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Welche Aufträge gibt man in den Pool?")
active = P.active_carriers(solved)
st.markdown(
    f"""
Jeder Spediteur mit Aufträgen gibt **genau einen** Stopp ab. Die abgegebenen Stopps werden danach bestmöglich auf alle Fahrzeuge verteilt (alle Zuordnungen, Touren exakt);
die übrigen Stopps bleiben bei ihrem Spediteur. Verglichen werden Auswahlregeln, die ein Spediteur selbst anwenden kann:
**teuerster** (der Stopp, dessen Wegnahme die eigene Tour am meisten verkürzt: die Wirkung auf den Gewinn), und vier **geografische** Lesarten (Nähe zu einem fremden Depot, Nähe zu einem
fremden Stopp, Abstand zum eigenen Depot, Abstand zu den eigenen anderen Stopps), dazu **zufällig** (Erwartungswert über alle Wahlen) und **bestmöglich** (alle {a.settings.per ** len(active)} Wahlen probiert).
"""
)
if a.settings.per == 1:
    st.info("Mit einem Stopp je Spediteur gibt es nichts zu wählen: Jeder gibt seinen einzigen Stopp ab, alle Regeln liefern dasselbe. Mehr Stopps je Spediteur einstellen.")
else:
    picks, sav = _pool_instance(a.settings)
    st.markdown("**Beispielspur in dieser Instanz:** welchen eigenen Stopp (Nummer ab 1) gibt jeder Spediteur ab, und was spart die Allianz damit?")
    st.dataframe(
        [{"Regel": C.CRITERIA_LABEL[c], **{f"Spediteur {i + 1}": picks[c][k] for k, i in enumerate(active)}, "Ersparnis": 100 * sav[c]} for c in C.CRITERIA_RULES]
        + [{"Regel": C.CRITERIA_LABEL[c], **{f"Spediteur {i + 1}": "–" for i in active}, "Ersparnis": 100 * sav[c]} for c in ("zufall", "bestmoeglich")],
        hide_index=True, width="stretch", column_config={"Ersparnis": st.column_config.NumberColumn(format="%.1f %%")},
    )
    if a.settings.per == 2:
        st.caption("Bei 2 Stopps je Spediteur sind beide Stopps gleich weit von den eigenen anderen Stopps entfernt: Die Regel „am weitesten von den eigenen anderen Stopps“ hat dann einen Gleichstand und wählt den ersten Stopp.")
    key3 = (settings.n, a.settings.per, a.settings.idle, settings.spread, settings.ninst)
    res3 = _run_experiment(
        "pool", key3, "🧪 Experiment 3 rechnen",
        lambda: E.pool_ensemble(settings.n, a.settings.per, settings.spread, a.settings.idle, _seeds(settings.ninst)),
        f"Rechne {settings.ninst} Instanzen mit allen {a.settings.per ** len(active)} Wahlen …",
    )
    if res3:
        cols, secs = res3
        rows3 = E.pool_rows(cols)
        st.plotly_chart(build_pool(rows3), width="stretch", key="chart_pool")
        st.dataframe(
            [{"Regel": C.CRITERIA_LABEL[r["criterion"]], "Mittel": r["mean"], "Median": r["median"], "erreicht die bestmögliche Wahl in": f"{r['optimal']} von {r['count']}"} for r in rows3],
            hide_index=True, width="stretch", column_config={k: st.column_config.NumberColumn(format="%.1f %%") for k in ("Mittel", "Median")},
        )
        geo = E.best_geo_name(rows3)
        kind, text = E.verdict_pool(rows3, E.compare_wins(cols, geo, "teuerster"), geo)
        getattr(st, kind)(text)
        st.markdown("**Alle vier geografischen Lesarten gegen den teuersten Auftrag** (Zahl der Instanzen: besser, gleich, schlechter):")
        st.dataframe(
            [{"Geografische Lesart": C.CRITERIA_LABEL[g], "besser": w, "gleich": e, "schlechter": l} for g in C.CRITERIA_GEO for (w, e, l) in [E.compare_wins(cols, g, "teuerster")]],
            hide_index=True, width="stretch",
        )
        st.caption(f"Rechenzeit {de(secs, 1)} s.")
    with st.expander("Was die Literatur sagt, und was hier nachgebaut ist"):
        st.markdown(
            """
Gansterer und Hartl schreiben im Newsletter (IFORS News, Dezember 2016) und im Überblick (arXiv 1706.05254), die besten Kriterien zur Bewertung von Aufträgen für den Pool
nähmen **geografische Aspekte** in den Blick und schlügen **reine Gewinnstrategien** deutlich. Im Überblick steht dazu eine Abbildung, in der ein Spediteur nach dem **Grenzgewinn** wählt
und zwei andere **geografische Informationen** einbeziehen. Eine genaue Definition der Kriterien steht in diesen beiden Texten nicht; sie verweisen auf einen Aufsatz, der hier **nicht gelesen** ist.

Darum sind hier **vier Lesarten von „geografisch“** gebaut und **eine Lesart von „Gewinn“** (die Entlastung der eigenen Tour). Ob die Aussage der Literatur in diesem Modell gilt,
zeigt die Zählung oben, und sie gilt nur für diese Lesarten in diesem Modell (Fahrzeuge ohne Kapazität, ein Stopp je Spediteur abgegeben, keine Preise, keine Auktion).
Die Literatur nennt außerdem das Team-Orienteering-Problem (Aufträge, die nicht in der besten Tour liegen, kommen in den Pool); es braucht Gewinne je Auftrag und begrenzte Fahrzeuge und ist hier **nicht nachgebaut**.
            """
        )

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Was bringt jedes weitere Mitglied?")
per4 = max(C.PER_MIN, min(a.settings.per, C.MAX_STOPS // C.N_MAX))
st.caption(
    f"Zahl der Spediteure von {C.N_MIN} bis {C.N_MAX}, jeder mit {per4} Stopps (bei {C.N_MAX} Spediteuren sind höchstens {C.MAX_STOPS // C.N_MAX} Stopps je Spediteur möglich), "
    f"{spread_text}, {settings.ninst} feste Instanzen je Größe. Mit jedem Mitglied kommen ein Depot, ein Fahrzeug und {per4} Stopps dazu."
)
key4 = (per4, settings.spread, settings.ninst)
res4 = _run_experiment(
    "members", key4, "🧪 Experiment 4 rechnen",
    lambda: E.sweep_members(per4, settings.spread, _seeds(settings.ninst)),
    f"Rechne {(C.N_MAX - C.N_MIN + 1) * settings.ninst} Instanzen exakt …",
)
if res4:
    rows4, secs = res4
    st.plotly_chart(build_members(rows4), width="stretch", key="chart_members")
    st.dataframe(
        [{"Spediteure": r["n"], "Instanzen": r["count"], "Mittel": r["mean"], "Median": r["median"], "Zuwachs gegenüber dem Vorgänger (Prozentpunkte)": r["delta"]} for r in rows4],
        hide_index=True, width="stretch",
        column_config={"Mittel": st.column_config.NumberColumn(format="%.1f %%"), "Median": st.column_config.NumberColumn(format="%.1f %%"),
                       "Zuwachs gegenüber dem Vorgänger (Prozentpunkte)": st.column_config.NumberColumn(format="%.1f")},
    )
    kind, text = E.verdict_members(rows4)
    getattr(st, kind)(text)
    st.caption(f"Rechenzeit {de(secs, 1)} s.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein Fahrzeug ohne Kapazität je Spediteur** | Mit Kapazitäten, Zeitfenstern oder Abhol-und-Liefer-Paaren ändern sich Tourkosten und Koalitionswerte; hier nicht gemessen. | vrp-demo (Tourenplanung mit Kapazitäten) |
| **Alles exakt, bis {C.MAX_STOPS} Stopps** | Der Aufwand wächst mit $3^M$ je Koalition; darüber braucht es Heuristiken, die hier fehlen. | – |
| **Eine Zentrale kennt alle Stopps und Kosten** | Spediteure geben Informationen ungern preis; ohne volle Information wird die beste Verteilung nicht erreicht. | auction-demo (kombinatorische Auktion), task-swap-demo (Auftragstausch) |
| **Jeder gibt genau einen Stopp ab** | Die Zahl der abgegebenen Stopps ist selbst eine Entscheidung; hier nicht untersucht. | – |
| **Die Ersparnis wird nur verteilt, nicht bezahlt** | Zahlungen, Anreize und strategisches Verhalten bei der Auswahl fehlen. | – |
| **Stopps streuen normalverteilt um das Depot** | Echte Kundenlagen sind geklumpt; die Streuungsstufen sind eine Stellvertreter-Größe für die Überlappung. | – |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        f"""
**Annahmen**
- Spediteure mit je einem Depot und einem Fahrzeug ohne Kapazität; Stopps und Depots im Einheitsquadrat, euklidische Entfernungen; Kosten sind Strecken.
- In der Allianz darf jedes Fahrzeug jeden Stopp der Mitglieder fahren; alle Rechnungen sind exakt (Held-Karp und Verteilung über alle Zuordnungen), nichts ist geschätzt.
- Die Stopps liegen normalverteilt um das eigene Depot und sind **nicht auf das Gebiet beschnitten**; bei Streuung {C.SPREAD_MAX} % liegen sie gleichverteilt im Einheitsquadrat.

**Grenzen**
- Höchstens {C.MAX_STOPS} Stopps insgesamt und {C.N_MAX} Spediteure. Keine Kapazitäten, Zeitfenster, Abhol-und-Liefer-Paare, keine mehreren Fahrzeuge je Spediteur, keine Zahlungen zwischen Spediteuren.
- Die Ersparnis ist die der besten gemeinsamen Verteilung bei voller Information; in diesem Modell ist das eine **Obergrenze** der Ersparnis jedes Verfahrens, das Stopps auf die Fahrzeuge verteilt.
- Der Kern ist nur eine Kennzahl; die Aufteilungstheorie steht in kern-demo.

**Bekannte Probleme und Unsicherheiten**
- Die Experimente laufen über feste Zufallsinstanzen (nach Einstellung 10 bis {C.NINST_MAX}); die Mittel schwanken mit dieser Stichprobe. Die Zahlen der README stammen aus größeren oder anderen festen Stichproben und können davon abweichen.
- Die geografischen Auswahlregeln sind **eigene Lesarten** nach Newsletter und Überblick; die Definitionen der Literatur sind nicht gelesen.
- Die Anzeige wurde im Browser an Presets und am Regler-Maximum geprüft, nicht auf jedem Gerät und nicht mit Touch-Bedienung.
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Spediteure mit Depots und Stopps; gesucht sind die Kosten jeder Koalition, daraus die Ersparnis, eine Aufteilung und die Wirkung der Auswahlregel.

| Symbol | Bedeutung |
|---|---|
| $N = \{1, \dots, n\}$ | Menge der Spediteure |
| $P$, $M = \lvert P \rvert$ | Menge aller Stopps und ihre Zahl |
| $P_i \subseteq P$ | die Stopps von Spediteur $i$ |
| $t_i(A)$ | Länge der kürzesten Rundtour ab Depot $i$ über die Stopp-Menge $A$ |
| $C(S)$ | Kosten der Koalition $S \subseteq N$ |
| $x_i$ | Kostenanteil von Spediteur $i$ |

**Tourkosten (Held-Karp).** Mit $f(A, j)$ als kürzestem Weg vom Depot über genau die Stopps in $A$, der in $j \in A$ endet:
$$f(\{j\}, j) = d(\text{Depot}, j)$$
$$f(A, j) = \min_{k \in A \setminus \{j\}} \left( f(A \setminus \{j\}, k) + d(k, j) \right)$$
$$t_i(A) = \min_{j \in A} \left( f(A, j) + d(j, \text{Depot}) \right), \qquad t_i(\emptyset) = 0$$

**Koalitionskosten.** Die Stopps der Mitglieder werden bestmöglich auf deren Fahrzeuge verteilt:
$$C(S) = \min \left\{ \sum_{i \in S} t_i(A_i) \;:\; (A_i)_{i \in S} \text{ ist eine Zerlegung von } \bigcup_{i \in S} P_i \right\}$$

**Ersparnis.**
$$\text{Ersparnis} = 1 - \frac{C(N)}{\sum_{i \in N} C(\{i\})}$$

**Shapley-Kostenanteil.**
$$x_i^{\text{Sh}} = \sum_{S \subseteq N \setminus \{i\}} \frac{\lvert S \rvert! \, (n - \lvert S \rvert - 1)!}{n!} \left( C(S \cup \{i\}) - C(S) \right)$$

**Proportionale Aufteilung.**
$$x_i^{\text{prop}} = C(N) \cdot \frac{C(\{i\})}{\sum_{j \in N} C(\{j\})}$$

**Kern und Least-Core-Wert.** Der Kern ist die Menge der Aufteilungen $x$ mit $\sum_i x_i = C(N)$ und $\sum_{i \in S} x_i \le C(S)$ für alle Koalitionen $S$. Das lineare Programm
$$\varepsilon^* = \min \left\{ \varepsilon \;:\; \sum_{i \in S} x_i \le C(S) + \varepsilon \;\; \forall\, \emptyset \ne S \subsetneq N, \;\; \sum_{i \in N} x_i = C(N) \right\}$$
zeigt: Der Kern ist genau dann nicht leer, wenn $\varepsilon^* \le 0$.

**Auswahl für den Pool.** Gibt Spediteur $i$ den Stopp $o_i \in P_i$ ab, wird $O = \{o_i\}$ bestmöglich auf alle Fahrzeuge verteilt:
$$K(o) = \min_{\sigma : O \to N} \sum_{c \in N} t_c \left( \left( P_c \setminus O \right) \cup \sigma^{-1}(c) \right)$$
Die bestmögliche Wahl minimiert $K(o)$ über alle $\prod_i \lvert P_i \rvert$ Kombinationen; die Regeln wählen $o_i$ aus lokalen Größen (Tourverkürzung $t_i(P_i) - t_i(P_i \setminus \{p\})$ oder Entfernungen).

**Literatur** (nur gelesen, was hier steht): M. Gansterer, R. F. Hartl: *Collaborative vehicle routing: a survey*, arXiv:1706.05254 (2017; nur die arXiv-Fassung gelesen).
M. Gansterer, R. F. Hartl: *Combinatorial Auctions in Collaborative Vehicle Routing*, IFORS News (Dezember 2016, Beitrag in der Reihe Tutorials; nur dieser Beitrag gelesen).
Quellen zu Held-Karp, Shapley-Wert und Bondareva-Shapley-Satz sind hier **nicht geprüft** und deshalb nicht zitiert.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
