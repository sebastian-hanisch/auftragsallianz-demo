"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import aal_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def carrier_color(i: int) -> str:
    return C.COLOR_CARRIER[i % len(C.COLOR_CARRIER)]


def build_map(inst, route_list):
    """Karte: Depots (Quadrate) und Stopps (Kreise) in der Farbe ihres Spediteurs; Linien = die Touren der besten gemeinsamen Verteilung, in der Farbe des Fahrzeugs.

    route_list: [(Fahrzeug, [Stopp-Indizes in Fahrtreihenfolge])]. Gleicher Maßstab über scaleanchor (nur mit autorange, DEMO-PLAYBOOK §3)."""
    fig = go.Figure()
    for c, order in route_list:
        pts = [inst.depots[c]] + [inst.stops[j] for j in order] + [inst.depots[c]]
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines", line=dict(color=carrier_color(c), width=2.5), hoverinfo="skip",
                                 name=f"Tour Fahrzeug {c + 1}", showlegend=False))
    for i in range(inst.n):
        mine = inst.stops_of(i)
        if mine:
            fig.add_trace(go.Scatter(x=[inst.stops[j][0] for j in mine], y=[inst.stops[j][1] for j in mine], mode="markers",
                                     marker=dict(color=carrier_color(i), size=11, line=dict(color="white", width=1)), name=f"Stopps Spediteur {i + 1}",
                                     hovertemplate=f"Spediteur {i + 1}, Stopp %{{text}}<extra></extra>", text=[str(j + 1) for j in mine]))
        fig.add_trace(go.Scatter(x=[inst.depots[i][0]], y=[inst.depots[i][1]], mode="markers", marker=dict(color=carrier_color(i), size=16, symbol="square",
                                 line=dict(color="black", width=1.5)), name=f"Depot Spediteur {i + 1}", hovertemplate=f"Depot Spediteur {i + 1}<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.05), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_overlap(rows):
    """Experiment 1: Mittel und Median der Ersparnis je Streuungsstufe (Balken = Mittel, Rauten = Median)."""
    labels = [("gleichverteilt" if r["spread"] >= C.SPREAD_MAX else f"{r['spread']} %") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["mean"] for r in rows], name="Mittel", marker_color=C.COLOR_LINE, hovertemplate="%{x}: Mittel %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=[r["median"] for r in rows], mode="markers", name="Median", marker=dict(symbol="diamond", size=12, color="#d98e04"),
                             hovertemplate="%{x}: Median %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Streuung der Stopps um das eigene Depot", type="category")
    fig.update_yaxes(title_text="Ersparnis (%)")
    return _base(fig)


def build_shares(a_alone, shapley, proportional):
    """Experiment 2: Ersparnis je Spediteur (Alleinkosten minus Kostenanteil, in Kosteneinheiten) nach Shapley und proportional."""
    labels = [f"Spediteur {i + 1}" for i in range(len(a_alone))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[float(a - s) for a, s in zip(a_alone, shapley)], name="Shapley", marker_color="#2F6B65",
                         hovertemplate="%{x}: %{y:.3f}<extra>Shapley</extra>"))
    fig.add_trace(go.Bar(x=labels, y=[float(a - p) for a, p in zip(a_alone, proportional)], name="proportional", marker_color="#d98e04",
                         hovertemplate="%{x}: %{y:.3f}<extra>proportional</extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Ersparnis des Spediteurs (Kosteneinheiten)")
    fig.update_layout(barmode="group")
    return _base(fig, 320)


def build_pool(rows):
    """Experiment 3: Mittel der Ersparnis je Auswahlkriterium (waagerechte Balken, bestmögliche Wahl und Zufall als Rahmen)."""
    labels = [C.CRITERIA_LABEL[r["criterion"]] for r in rows][::-1]
    colors = [("#7a5195" if r["criterion"] in ("zufall", "bestmoeglich") else ("#2F6B65" if r["criterion"] == "teuerster" else C.COLOR_LINE)) for r in rows][::-1]
    fig = go.Figure(go.Bar(x=[r["mean"] for r in rows][::-1], y=labels, orientation="h", marker_color=colors, hovertemplate="%{y}: Mittel %{x:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Ersparnis gegen \"jeder fährt allein\" (Mittel, %)")
    fig.update_yaxes(type="category")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_members(rows):
    """Experiment 4: Mittel der Ersparnis über die Zahl der Spediteure."""
    xs = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["mean"] for r in rows], mode="lines+markers", name="Mittel", line=dict(color=C.COLOR_LINE, width=3), hovertemplate="%{x} Spediteure: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["median"] for r in rows], mode="markers", name="Median", marker=dict(symbol="diamond", size=11, color="#d98e04"), hovertemplate="%{x} Spediteure: Median %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Zahl der Spediteure", tickmode="array", tickvals=xs)
    fig.update_yaxes(title_text="Ersparnis (%)")
    return _base(fig)
