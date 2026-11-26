"""Persistent homology of the karate club network filtered by eigenvector centrality.

Filtration (sublevel set of the centrality function on the graph):

* vertex ``v`` is born at height ``c(v)``: it enters once the threshold
  reaches its centrality;
* edge ``(u, v)`` is born at ``max(c(u), c(v))``: it enters once both
  endpoints are present.

Low-centrality nodes enter first, and the hubs enter last.
``superlevel=True`` reverses the order so hubs enter first; it does this by
negating ``c``.

By default the complex is the graph itself (0- and 1-simplices), as specified.
With no 2-simplices nothing can fill a loop, so every H1 class lives forever.
``max_dim=None`` uses the full clique (flag) complex instead: each clique is
born when its last vertex is, and loops that cliques fill die.
"""

from pathlib import Path

import gudhi
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

INK = "#191919"
INK_2 = "#666666"
GRID = "#e6e6e6"
H0_COLOR = "#257de0"  # categorical slot 1 (blue)
H1_COLOR = "#f47230"  # categorical slot 2 (orange)


def centrality_simplex_tree(G, values, max_dim=1, superlevel=False):
    """Build a gudhi SimplexTree with the vertex-centrality filtration.

    Parameters
    ----------
    G : networkx.Graph
    values : dict
        Node -> filtration value (e.g. eigenvector centrality).
    max_dim : int or None
        1 for the graph (vertices + edges); k >= 2 adds cliques up to dimension k;
        None adds every clique (the full clique complex). Truncating below the
        largest clique leaves hollow cliques that show up as spurious voids.
    superlevel : bool
        If True, filter by -value so the highest-centrality nodes enter first.
    """
    sign = -1.0 if superlevel else 1.0
    f = {v: sign * float(values[v]) for v in G}
    st = gudhi.SimplexTree()
    for v in G:
        st.insert([v], filtration=f[v])
    for u, v in G.edges():
        st.insert([u, v], filtration=max(f[u], f[v]))
    if max_dim is None:
        max_dim = max(len(c) for c in nx.find_cliques(G)) - 1
    if max_dim > 1:
        # Each higher simplex gets the max filtration of its faces.
        st.expansion(max_dim)
    st.make_filtration_non_decreasing()
    return st


def persistence_summary(st):
    """Compute persistence and return per-dimension intervals plus generator pairs."""
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
    dims = range(st.dimension() + 1)
    intervals = {d: st.persistence_intervals_in_dimension(d) for d in dims}
    pairs = st.persistence_pairs()  # (birth simplex, death simplex) lists
    return intervals, pairs


def betti_curve(intervals, thresholds):
    if len(intervals) == 0:
        return np.zeros_like(thresholds)
    b, d = intervals[:, 0], intervals[:, 1]
    return np.array([np.sum((b <= t) & (t < d)) for t in thresholds])


def h0_bars_with_vertices(st, pairs):
    """H0 bars labeled by the vertex that creates each component."""
    bars = []
    for birth_s, death_s in pairs:
        if len(birth_s) != 1:
            continue
        v = birth_s[0]
        b = st.filtration([v])
        d = st.filtration(death_s) if death_s else np.inf
        bars.append((v, b, d))
    return sorted(bars, key=lambda x: (x[2] - x[1], -x[1]), reverse=True)


def plot_tda(G, st, intervals, pairs, path, title):
    fvals = np.array(sorted({st.filtration([v]) for v in G}))
    lo, hi = fvals.min(), fvals.max()
    pad = 0.05 * (hi - lo)
    inf_y = hi + pad  # where infinite bars are drawn

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2),
                             gridspec_kw={"width_ratios": [1.25, 1, 1.1]})

    # (a) H0 barcode, labeled by creating vertex
    ax = axes[0]
    bars = [bar for bar in h0_bars_with_vertices(st, pairs) if bar[2] > bar[1]]
    for i, (v, b, d) in enumerate(bars):
        end = inf_y if np.isinf(d) else d
        ax.plot([b, end], [i, i], color=H0_COLOR, linewidth=3, solid_capstyle="round")
        if np.isinf(d):
            ax.annotate("", xy=(inf_y + pad * 0.6, i), xytext=(inf_y, i),
                        arrowprops=dict(arrowstyle="->", color=H0_COLOR, lw=1.5))
    ax.set_yticks(range(len(bars)), [f"node {v}" for v, _, _ in bars], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(lo - pad, inf_y + 1.5 * pad)
    ax.set_xlabel("Filtration value (eigenvector centrality)")
    ax.set_title("H0 barcode: components, by birth node", loc="left", fontsize=12)
    # gudhi omits zero-length pairs: those vertices join an existing component on arrival.
    n_zero = G.number_of_nodes() - len(bars)
    ax.text(0.98, 0.02, f"{n_zero} nodes join an existing component on arrival", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=INK_2)

    # (b) persistence diagram
    ax = axes[1]
    ax.plot([lo - pad, inf_y], [lo - pad, inf_y], color=INK_2, lw=1, ls="--")
    ax.axhline(inf_y, color=GRID, lw=1)
    ax.text(lo - pad, inf_y, " ∞", va="bottom", color=INK_2, fontsize=10)
    for d, color, marker in ((0, H0_COLOR, "o"), (1, H1_COLOR, "s")):
        iv = intervals.get(d, np.empty((0, 2)))
        if len(iv) == 0:
            continue
        deaths = np.where(np.isinf(iv[:, 1]), inf_y, iv[:, 1])
        ax.scatter(iv[:, 0], deaths, s=60, color=color, marker=marker,
                   edgecolors="white", linewidths=1.5, alpha=0.85,
                   label=f"H{d} ({len(iv)} classes)", zorder=3)
    ax.set_xlim(lo - pad, inf_y + pad)
    ax.set_ylim(lo - pad, inf_y + 2 * pad)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    ax.set_title("Persistence diagram", loc="left", fontsize=12)

    # (c) Betti curves
    ax = axes[2]
    t = np.linspace(lo - pad, hi + pad, 600)
    for d, color in ((0, H0_COLOR), (1, H1_COLOR)):
        iv = intervals.get(d, np.empty((0, 2)))
        y = betti_curve(iv, t)
        ax.step(t, y, where="post", color=color, lw=2)
        ax.text(t[-1], y[-1], f"  β{d} = {y[-1]}", color=INK, va="center", fontsize=10)
    ax.set_xlim(lo - pad, hi + 3.5 * pad)
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Betti number")
    ax.set_title("Betti curves", loc="left", fontsize=12)

    for ax in axes:
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, length=0)
        ax.xaxis.label.set_color(INK_2)
        ax.yaxis.label.set_color(INK_2)

    fig.suptitle(title, x=0.01, ha="left", fontsize=15, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def report(name, st, intervals, pairs):
    print(f"\n=== {name} ===")
    print(f"simplices: {st.num_simplices()}  (vertices {st.num_vertices()}, dim {st.dimension()})")
    for d, iv in intervals.items():
        n_inf = int(np.isinf(iv[:, 1]).sum()) if len(iv) else 0
        n_pos = int(((iv[:, 1] - iv[:, 0]) > 0).sum()) if len(iv) else 0
        print(f"H{d}: {len(iv)} intervals, {n_inf} infinite, {n_pos} with positive length")
    print("Longest finite H0 bars (birth node: birth -> death):")
    for v, b, d in [x for x in h0_bars_with_vertices(st, pairs) if np.isfinite(x[2])][:5]:
        print(f"  node {v:2d}: {b:.4f} -> {d:.4f}  (persistence {d - b:.4f})")
    print("Betti numbers of full complex:", st.betti_numbers())


if __name__ == "__main__":
    G = nx.karate_club_graph()
    ec = nx.eigenvector_centrality(G, max_iter=1000)
    out = Path("figures")
    out.mkdir(exist_ok=True)

    configs = [
        ("graph_sublevel", 1, False,
         "Eigenvector-centrality filtration on the karate club graph (vertices + edges)"),
        ("clique_sublevel", None, False,
         "Same filtration on the full clique complex (cliques fill loops)"),
    ]
    for tag, max_dim, sup, title in configs:
        st = centrality_simplex_tree(G, ec, max_dim=max_dim, superlevel=sup)
        intervals, pairs = persistence_summary(st)
        report(tag, st, intervals, pairs)
        plot_tda(G, st, intervals, pairs, out / f"tda_eigenvector_{tag}.png", title)
    print("\nSaved figures to", out.resolve())
