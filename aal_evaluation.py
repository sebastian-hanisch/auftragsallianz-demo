"""Kennzahlen, Verdikt-Logik und Sweeps.

Alle Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen und Zahlen
an handgesetzten Zeilen. Alle Sweeps laufen über feste Instanzen (`SWEEP_SEEDS`), nie über einen einzelnen Seed."""

from dataclasses import dataclass
from statistics import median

import numpy as np

import aal_algorithm as A
import aal_constants as C
import aal_game as G
import aal_pool as P
from aal_scenario import make_instance

def de(value: float, digits: int = 1, signed: bool = False) -> str:
    """Zahl mit Dezimalkomma wie auf der Website (die Anzeige im Browser folgt der Locale nur in Zahlenspalten, nicht im Text); signed setzt ein Pluszeichen vor positive Werte."""
    s = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return s.replace(".", ",")


ZERO = 1e-9   # Ersparnis bis zu dieser Höhe (Anteil) zählt als "keine Ersparnis"


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    per: int = C.DEFAULT_PER
    spread: int = C.DEFAULT_SPREAD
    idle: int = C.DEFAULT_IDLE
    ninst: int = C.DEFAULT_NINST
    seed: int = C.DEFAULT_SEED


def effective_settings(s: Settings) -> Settings:
    """Bringt Einstellungen auf zulässige Werte: Spediteure ohne Aufträge höchstens n - 1, Stopps je Spediteur nach `C.max_per`."""
    idle = max(C.IDLE_MIN, min(s.idle, s.n - 1))
    per = max(C.PER_MIN, min(s.per, C.max_per(s.n, idle)))
    return Settings(s.n, per, s.spread, idle, s.ninst, s.seed)


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    solved: A.Solved
    shapley: np.ndarray            # Kostenanteile
    proportional: np.ndarray
    epsilon: float                 # Least-Core-Wert (Kern nicht leer genau dann, wenn <= 0)
    core_empty: bool
    shapley_in_core: bool
    proportional_in_core: bool
    assignment: list[int]          # Fahrzeug je Stopp in der besten gemeinsamen Verteilung


def saving_percent(grand: float, alone: float) -> float:
    """Ersparnis in Prozent der Alleinkosten; ohne Alleinkosten gibt es nichts zu sparen."""
    return 0.0 if alone <= 0 else 100.0 * (1.0 - grand / alone)


def analyse(settings: Settings) -> Analysis:
    s = effective_settings(settings)
    inst = make_instance(s.n, s.per, s.spread, s.idle, s.seed)
    solved = A.solve(inst)
    vals, n = solved.values, inst.n
    sh, pr = G.shapley_costs(vals, n), G.proportional_costs(vals, n)
    eps = G.least_core_epsilon(vals, n)
    assign = A.optimal_assignment(solved)
    if not A.check_feasible(inst, assign):
        raise RuntimeError("Die Zuordnung der Stopps ist unzulässig")
    return Analysis(s, solved, sh, pr, eps, eps > C.CORE_TOL * max(1.0, solved.grand), G.in_core(sh, vals, n), G.in_core(pr, vals, n), assign)


def routes(a: Analysis) -> list[tuple[int, list[int]]]:
    """Die Touren der besten gemeinsamen Verteilung: (Fahrzeug, Stopps in Fahrtreihenfolge) für jedes Fahrzeug mit mindestens einem Stopp."""
    inst = a.solved.inst
    out = []
    for c in range(inst.n):
        mask = sum(1 << j for j, v in enumerate(a.assignment) if v == c)
        if mask:
            out.append((c, A.tour_order(inst.depots[c], inst.stops, mask)))
    return out


# --- Ensembles (Mehr-Seed-Verteilungen) ------------------------------------------------------------------------------------------------------

def ensemble(n: int, per: int, spread: int, idle: int, seeds) -> dict:
    """Ersparnis, Kern und Aufteilung über feste Instanzen. Gibt Listen je Instanz zurück (gleiche Reihenfolge wie `seeds`)."""
    savings, empty, sh_in, pr_in = [], 0, 0, 0
    for seed in seeds:
        solved = A.solve(make_instance(n, per, spread, idle, seed))
        savings.append(solved.saving)
        vals = solved.values
        empty += G.core_is_empty(vals, n)
        sh_in += G.in_core(G.shapley_costs(vals, n), vals, n)
        pr_in += G.in_core(G.proportional_costs(vals, n), vals, n)
    return {"savings": savings, "core_empty": empty, "shapley_in_core": sh_in, "proportional_in_core": pr_in, "count": len(savings)}


def sweep_spread(n: int, per: int, idle: int, seeds, stages=C.SPREAD_STAGES) -> list[dict]:
    """Experiment 1: Ersparnis über die Streuung; Mittel, Median, Zahl der Instanzen ohne Ersparnis, Kern- und Aufteilungszählung."""
    rows = []
    for spread in stages:
        e = ensemble(n, per, spread, idle, seeds)
        sv = e["savings"]
        rows.append({
            "spread": spread, "count": e["count"], "mean": 100 * float(np.mean(sv)), "median": 100 * float(median(sv)), "min": 100 * min(sv), "max": 100 * max(sv),
            "zero": sum(1 for v in sv if v <= ZERO), "core_empty": e["core_empty"], "shapley_in_core": e["shapley_in_core"], "proportional_in_core": e["proportional_in_core"],
        })
    return rows


def sweep_members(per: int, spread: int, seeds, sizes=range(C.N_MIN, C.N_MAX + 1)) -> list[dict]:
    """Experiment 4: Ersparnis über die Zahl der Spediteure (je Spediteur `per` Stopps, jeder mit Aufträgen); `delta` = Zuwachs des Mittels gegenüber dem Vorgänger in Prozentpunkten."""
    rows, prev = [], None
    for n in sizes:
        sv = ensemble_savings_only(n, per, spread, 0, seeds)
        mean = 100 * float(np.mean(sv))
        rows.append({"n": n, "count": len(sv), "mean": mean, "median": 100 * float(median(sv)), "delta": None if prev is None else mean - prev})
        prev = mean
    return rows


def ensemble_savings_only(n: int, per: int, spread: int, idle: int, seeds) -> list[float]:
    return [A.solve(make_instance(n, per, spread, idle, seed)).saving for seed in seeds]


def pool_ensemble(n: int, per: int, spread: int, idle: int, seeds) -> dict:
    """Experiment 3: Ersparnis je Auswahlkriterium über feste Instanzen. Gibt je Kriterium die Liste der Ersparnisse (Anteil) je Instanz zurück."""
    cols: dict[str, list[float]] = {c: [] for c in C.CRITERIA}
    for seed in seeds:
        solved = A.solve(make_instance(n, per, spread, idle, seed))
        for name, value in P.criterion_savings(solved).items():
            cols[name].append(value)
    return cols


def pool_rows(cols: dict) -> list[dict]:
    """Zeilen der Auswahl-Tabelle: Mittel, Median, Zahl der Instanzen, in denen das Kriterium die bestmögliche Wahl erreicht (innerhalb ZERO)."""
    best = cols["bestmoeglich"]
    rows = []
    for name in C.CRITERIA:
        v = cols[name]
        rows.append({
            "criterion": name, "mean": 100 * float(np.mean(v)), "median": 100 * float(median(v)), "count": len(v),
            "optimal": sum(1 for x, b in zip(v, best) if x >= b - ZERO),
        })
    return rows


def compare_wins(cols: dict, a: str, b: str) -> tuple[int, int, int]:
    """Zählung (a besser, gleich, b besser) über die Instanzen; gleich heißt Unterschied bis ZERO."""
    wins = ties = losses = 0
    for x, y in zip(cols[a], cols[b]):
        if x > y + ZERO:
            wins += 1
        elif y > x + ZERO:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def best_geo_name(rows: list[dict]) -> str:
    """Das geografische Kriterium mit dem größten Mittel (Gleichstand: das erste in C.CRITERIA_GEO)."""
    by = {r["criterion"]: r["mean"] for r in rows}
    return max(C.CRITERIA_GEO, key=lambda c: (by[c], -C.CRITERIA_GEO.index(c)))


# --- Verdikte (reine Funktionen) -------------------------------------------------------------------------------------------------------------

def verdict_saving(alone: float, grand: float) -> tuple[str, str]:
    """Verdikt der Kernfrage für die gewählte Instanz."""
    pct = saving_percent(grand, alone)
    if alone <= 0 or pct <= ZERO * 100:
        return "warning", f"Hier spart das Zusammenlegen nichts: gemeinsam {de(grand, 2)}, alle allein {de(alone, 2)}. Jeder Spediteur fährt allein genauso billig."
    if pct >= 20 - 1e-9:
        return "success", f"Zusammenlegen spart **{de(pct, 1)} %**: gemeinsam {de(grand, 2)} statt {de(alone, 2)}, wenn jeder allein fährt."
    if pct >= 5 - 1e-9:
        return "info", f"Zusammenlegen spart **{de(pct, 1)} %** ({de(grand, 2)} statt {de(alone, 2)})."
    return "info", f"Zusammenlegen spart nur **{de(pct, 1)} %** ({de(grand, 2)} statt {de(alone, 2)})."


def verdict_overlap(rows: list[dict]) -> tuple[str, str]:
    """Verdikt Experiment 1: Mittel der kleinsten und der größten Streuung, Zahl der Stufen mit steigendem Mittel, Instanzen ohne Ersparnis."""
    first, last = rows[0], rows[-1]
    rises = sum(1 for a, b in zip(rows, rows[1:]) if b["mean"] > a["mean"])
    steps = len(rows) - 1
    cnt = first["count"]
    return "info", (
        f"Bei Streuung {first['spread']} % spart das Zusammenlegen im Mittel {de(first['mean'], 1)} % (Median {de(first['median'], 1)} %, "
        f"{first['zero']} von {cnt} Instanzen ganz ohne Ersparnis), bei Streuung {last['spread']} % im Mittel {de(last['mean'], 1)} % "
        f"({last['zero']} von {cnt} ohne Ersparnis). Das Mittel steigt in {rises} von {steps} Stufen."
    )


def verdict_shares(a: Analysis) -> tuple[str, str]:
    """Verdikt Experiment 2 für die gewählte Instanz: liegen Shapley und proportionale Aufteilung im Kern?"""
    if a.core_empty:
        return "warning", f"Der Kern ist leer (Least-Core-Wert {de(a.epsilon, 3)} > 0): keine Aufteilung stellt jede Gruppe mindestens so gut wie allein."
    sh = "im Kern" if a.shapley_in_core else "außerhalb des Kerns"
    pr = "im Kern" if a.proportional_in_core else "außerhalb des Kerns"
    kind = "success" if (a.shapley_in_core and a.proportional_in_core) else "info"
    return kind, f"Der Kern ist nicht leer. In dieser Instanz liegt die Shapley-Aufteilung {sh}, die proportionale Aufteilung {pr}."


def verdict_shares_ensemble(rows: list[dict]) -> tuple[str, str]:
    """Verdikt Experiment 2 über die Instanzen (Zeilen aus `sweep_spread`): Zählung leerer Kerne und Aufteilungen im Kern über alle Stufen."""
    total = sum(r["count"] for r in rows)
    empty = sum(r["core_empty"] for r in rows)
    sh = sum(r["shapley_in_core"] for r in rows)
    pr = sum(r["proportional_in_core"] for r in rows)
    kind = "warning" if empty else "info"
    return kind, (f"Über alle {total} Instanzen dieses Experiments (alle Stufen zusammen) war der Kern in {empty} leer; die Shapley-Aufteilung lag in {sh}, "
                  f"die proportionale in {pr} im Kern.")


def verdict_pool(rows: list[dict], wins: tuple[int, int, int], geo: str) -> tuple[str, str]:
    """Verdikt Experiment 3: bestes geografisches Kriterium gegen den teuersten Auftrag, mit Zählung (besser, gleich, schlechter) über die Instanzen."""
    by = {r["criterion"]: r for r in rows}
    t, g = by["teuerster"], by[geo]
    w, e, l = wins
    cnt = t["count"]
    if g["mean"] > t["mean"]:
        head = "Hier liegt ein geografisches Kriterium vorn"
    elif g["mean"] < t["mean"] and t["mean"] - g["mean"] < 1.0:
        head = "Hier liegt der teuerste Auftrag nur knapp vor dem besten geografischen Kriterium (Abstand unter 1 Prozentpunkt, nahezu gleichauf)"
    elif g["mean"] < t["mean"]:
        head = "Hier liegt der teuerste Auftrag im Mittel vor jedem geografischen Kriterium"
    else:
        head = "Beide liegen gleichauf"
    return "info", (
        f"{head}: bestes geografisches Kriterium ({C.CRITERIA_LABEL[geo]}) im Mittel {de(g['mean'], 1)} %, teuerster Auftrag {de(t['mean'], 1)} %, "
        f"Zufall {de(by['zufall']['mean'], 1)} %, bestmöglich {de(by['bestmoeglich']['mean'], 1)} % ({cnt} Instanzen). "
        f"Das geografische Kriterium ist in {w} Instanzen besser, in {e} gleich und in {l} schlechter als der teuerste Auftrag."
    )


def verdict_members(rows: list[dict]) -> tuple[str, str]:
    """Verdikt Experiment 4: Mittel bei der kleinsten und größten Zahl der Spediteure und die Zuwächse je weiterem Mitglied."""
    first, last = rows[0], rows[-1]
    deltas = ", ".join(f"{de(r['delta'], 1, True)}" for r in rows[1:])
    shrinking = sum(1 for a, b in zip(rows[1:], rows[2:]) if b["delta"] < a["delta"])
    return "info", (
        f"Bei {first['n']} Spediteuren spart das Zusammenlegen im Mittel {de(first['mean'], 1)} %, bei {last['n']} sind es {de(last['mean'], 1)} % ({first['count']} Instanzen). "
        f"Zuwachs je weiterem Mitglied in Prozentpunkten: {deltas}. Der Zuwachs sinkt in {shrinking} von {max(0, len(rows) - 2)} Schritten (kleine Unterschiede, bei dieser Instanzzahl nicht abgesichert)."
    )
