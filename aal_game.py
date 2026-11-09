"""Kern, Teil 2: Aufteilung der Ersparnis (Kostenspiel mit C(S) aus `aal_algorithm`).

Kostenanteile x_i: Shapley-Wert (exakt aus den Koalitionswerten), proportional zu den Alleinkosten, und die Kennzahl "im Kern?" per linearem Programm (HiGHS).
Die Theorie der Aufteilung (Kern, Nukleolus, Zusatzbedingungen) steht in kern-demo und shapley-demo und wird hier nicht neu gebaut: der Kern dient nur als Kennzahl."""

import math

import numpy as np

import aal_constants as C


def shapley_costs(values: np.ndarray, n: int) -> np.ndarray:
    """Shapley-Kostenanteile aus den Koalitionswerten: x_i = Summe über S ohne i von |S|! (n-|S|-1)! / n! mal (C(S plus i) - C(S))."""
    x = np.zeros(n)
    for i in range(n):
        for s in range(1 << n):
            if (s >> i) & 1:
                continue
            size = bin(s).count("1")
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            x[i] += weight * (values[s | (1 << i)] - values[s])
    return x


def proportional_costs(values: np.ndarray, n: int) -> np.ndarray:
    """Kostenanteile proportional zu den Alleinkosten; sind alle Alleinkosten null, zahlt jeder gleich viel."""
    alone = np.array([values[1 << i] for i in range(n)])
    grand = values[(1 << n) - 1]
    total = alone.sum()
    return np.full(n, grand / n) if total <= 0 else grand * alone / total


def max_excess(x: np.ndarray, values: np.ndarray, n: int) -> float:
    """Größter Überschuss x(S) - C(S) über alle echten Teilkoalitionen S (nicht leer, nicht alle): <= 0 heißt, keine Gruppe zahlt mehr als allein."""
    best = -np.inf
    for s in range(1, (1 << n) - 1):
        members = [i for i in range(n) if (s >> i) & 1]
        best = max(best, float(x[members].sum() - values[s]))
    return best


def in_core(x: np.ndarray, values: np.ndarray, n: int) -> bool:
    """Liegt die Aufteilung im Kern? Effizient (Summe = C(alle)) und kein Überschuss über der Toleranz."""
    grand = float(values[(1 << n) - 1])
    tol = C.CORE_TOL * max(1.0, abs(grand))
    return abs(float(x.sum()) - grand) <= tol and max_excess(x, values, n) <= tol


def least_core_epsilon(values: np.ndarray, n: int) -> float:
    """Kleinstes eps, für das x(S) <= C(S) + eps für alle echten Teilkoalitionen und x(alle) = C(alle) lösbar ist (lineares Programm, HiGHS).

    Der Kern ist genau dann nicht leer, wenn eps <= 0."""
    from scipy.optimize import linprog

    rows, rhs = [], []
    for s in range(1, (1 << n) - 1):
        row = np.zeros(n + 1)
        for i in range(n):
            if (s >> i) & 1:
                row[i] = 1.0
        row[n] = -1.0
        rows.append(row)
        rhs.append(values[s])
    res = linprog(
        np.append(np.zeros(n), 1.0), A_ub=np.array(rows), b_ub=np.array(rhs),
        A_eq=[np.append(np.ones(n), 0.0)], b_eq=[values[(1 << n) - 1]], bounds=[(None, None)] * (n + 1), method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"Kern-LP nicht gelöst: {res.message}")
    return float(res.x[n])


def core_is_empty(values: np.ndarray, n: int) -> bool:
    """Ist der Kern leer? (eps oberhalb der Toleranz)"""
    grand = float(values[(1 << n) - 1])
    return least_core_epsilon(values, n) > C.CORE_TOL * max(1.0, abs(grand))
