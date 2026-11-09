"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Regler-Maximum (DEMO-PLAYBOOK §1c): Mit M = Zahl aller Stopps ist Held-Karp je Fahrzeug O(2^M * M^2), die Verteilung der Stopps auf die
Fahrzeuge einer Koalition O(3^M) je Koalition, bei n Spediteuren also 2^n - 1 Koalitionen. Darum gilt MAX_STOPS = 12 als harte Obergrenze der Gesamtzahl
(M <= 12: 3^12 = 531 441 Paare je Koalition); der Regler "Stopps je Spediteur" ist an die Zahl der Spediteure mit Aufträgen gebunden."""

N_MIN, N_MAX, DEFAULT_N = 2, 5, 4                    # Spediteure (jeder mit eigenem Depot und einem Fahrzeug)
PER_MIN, PER_MAX, DEFAULT_PER = 1, 6, 2              # Stopps je Spediteur mit Aufträgen (Obergrenze zusätzlich über MAX_STOPS)
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 2, 100, 15  # Streuung der Stopps um das eigene Depot in % der Gebietsbreite; SPREAD_MAX = gleichverteilt
IDLE_MIN, DEFAULT_IDLE = 0, 0                        # Spediteure ohne eigene Aufträge (nur Depot und Fahrzeug); Obergrenze n - 1
NINST_MIN, NINST_MAX, DEFAULT_NINST, NINST_STEP = 10, 60, 30, 10   # Zahl der Zufallsinstanzen in den Experimenten (erst auf Klick gerechnet)
SEED_MAX, DEFAULT_SEED = 999999, 35
MAX_STOPS = 12                                       # Gesamtzahl der Stopps, bis zu der alle Rechnungen exakt laufen
SWEEP_SEEDS = tuple(range(100000, 100030))           # feste Instanzen für die Sweeps: nie ein einzelner Seed als Befund
SPREAD_STAGES = (5, 15, 30, SPREAD_MAX)              # Stufen des Experiments "Ersparnis über die Streuung" (in %)
CORE_TOL = 1e-9                                      # relative Toleranz für "im Kern"

COLOR_CARRIER = ("#2F6B65", "#c0504d", "#4c78a8", "#d98e04", "#7a5195")   # je Spediteur eine Farbe (Index = Nummer)
COLOR_LINE = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.55)"

CRITERIA = ("zufall", "teuerster", "fremdes_depot", "fremder_stopp", "weit_vom_depot", "isoliert", "bestmoeglich")
CRITERIA_LABEL = {
    "zufall": "zufällig (Erwartung)",
    "teuerster": "teuerster (größte Entlastung der eigenen Tour)",
    "fremdes_depot": "nächster an einem fremden Depot",
    "fremder_stopp": "nächster an einem fremden Stopp",
    "weit_vom_depot": "am weitesten vom eigenen Depot",
    "isoliert": "am weitesten von den eigenen anderen Stopps",
    "bestmoeglich": "bestmöglich (alle Wahlen)",
}
CRITERIA_GEO = ("fremdes_depot", "fremder_stopp", "weit_vom_depot", "isoliert")   # die geografischen Auslegungen
CRITERIA_RULES = ("teuerster",) + CRITERIA_GEO                                    # die Regeln, die ein Spediteur selbst anwenden kann

PRESETS = {
    "Mittlere Streuung (Voreinstellung)": {"n": DEFAULT_N, "per": DEFAULT_PER, "spread": DEFAULT_SPREAD, "idle": 0, "ninst": DEFAULT_NINST, "seed": DEFAULT_SEED},
    "Eng um die Depots": {"n": DEFAULT_N, "per": DEFAULT_PER, "spread": 5, "idle": 0, "ninst": DEFAULT_NINST, "seed": DEFAULT_SEED},
    "Gleichverteilt im Gebiet": {"n": DEFAULT_N, "per": DEFAULT_PER, "spread": SPREAD_MAX, "idle": 0, "ninst": DEFAULT_NINST, "seed": DEFAULT_SEED},
    "Fünf Spediteure": {"n": 5, "per": 2, "spread": DEFAULT_SPREAD, "idle": 0, "ninst": DEFAULT_NINST, "seed": DEFAULT_SEED},
    "Nur ein Spediteur hat Aufträge": {"n": 4, "per": 4, "spread": 30, "idle": 3, "ninst": DEFAULT_NINST, "seed": 1},
}
PRESET_HELP = {
    "Mittlere Streuung (Voreinstellung)": "4 Spediteure mit je 2 Stopps, die Stopps streuen mit 15 % der Gebietsbreite um das eigene Depot.",
    "Eng um die Depots": "Die Stopps liegen dicht am eigenen Depot (Streuung 5 %): jeder Spediteur bedient sein eigenes Revier, es gibt kaum etwas zusammenzulegen.",
    "Gleichverteilt im Gebiet": "Die Stopps liegen im ganzen Gebiet, die Reviere überlappen vollständig (Streuung 100 % heißt gleichverteilt).",
    "Fünf Spediteure": "5 Spediteure mit je 2 Stopps: mehr Mitglieder, 10 Stopps, 31 Koalitionen.",
    "Nur ein Spediteur hat Aufträge": "Ein Spediteur mit 4 Stopps, drei weitere haben nur Depot und Fahrzeug: nur einer hat Aufträge, die anderen stellen Fahrzeuge bereit.",
}


def max_per(n: int, idle: int) -> int:
    """Obergrenze der Stopps je Spediteur mit Aufträgen: höchstens PER_MAX und höchstens MAX_STOPS Stopps insgesamt."""
    return max(PER_MIN, min(PER_MAX, MAX_STOPS // max(1, n - idle)))
