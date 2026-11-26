"""Visualize centrality measures for the karate club network.

Produces two figures in ``figures/``:

* ``karate_centrality_networks.png`` - one network panel per measure, nodes
  sized and shaded by that measure (min-max scaled within the panel).
  Circles are Mr. Hi's faction, squares the Officer's.
* ``karate_centrality_heatmap.png`` - nodes x measures, each column min-max
  scaled, nodes grouped by faction and sorted by degree.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

from karate_centrality import compute_karate_centralities

# Single-hue sequential ramp (light -> dark blue) and neutral chart inks.
BLUE_RAMP = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
INK = "#191919"
INK_2 = "#666666"
EDGE = "#d0d0d0"

LABELS = {
    "degree": "Degree",
    "betweenness": "Betweenness",
    "closeness": "Closeness",
    "eigenvector": "Eigenvector",
    "pagerank": "PageRank",
    "katz": "Katz",
    "harmonic": "Harmonic",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.edgecolor": EDGE,
})


def _minmax(s):
    return (s - s.min()) / (s.max() - s.min())


def plot_networks(G, df, path):
    pos = nx.kamada_kawai_layout(G)
    measures = list(LABELS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    hi = [n for n in G if df.loc[n, "club"] == "Mr. Hi"]
    off = [n for n in G if df.loc[n, "club"] != "Mr. Hi"]

    for ax, m in zip(axes.flat, measures):
        scaled = _minmax(df[m])
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=EDGE, width=0.8)
        for nodes, shape in ((hi, "o"), (off, "s")):
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, ax=ax, node_shape=shape,
                node_color=[scaled[n] for n in nodes], cmap=CMAP, vmin=0, vmax=1,
                node_size=[60 + 540 * scaled[n] for n in nodes],
                edgecolors="white", linewidths=1.5,
            )
        top = df[m].nlargest(3).index
        nx.draw_networkx_labels(
            G, pos, labels={n: str(n) for n in top}, ax=ax,
            font_size=8, font_color="white", font_weight="bold",
        )
        ax.set_title(LABELS[m], fontsize=13, color=INK, loc="left")
        ax.axis("off")

    # Last panel: legend for shape and color.
    ax = axes.flat[-1]
    ax.axis("off")
    ax.scatter([], [], marker="o", s=120, color=BLUE_RAMP[4], label="Mr. Hi faction")
    ax.scatter([], [], marker="s", s=120, color=BLUE_RAMP[4], label="Officer faction")
    ax.legend(loc="upper left", frameon=False, fontsize=11, labelcolor=INK_2)
    sm = plt.cm.ScalarMappable(cmap=CMAP)
    cax = ax.inset_axes([0.05, 0.35, 0.8, 0.06])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["lowest", "highest"])
    cb.outline.set_visible(False)
    cax.set_title("Centrality (scaled within panel)", fontsize=10, color=INK_2, loc="left")
    ax.text(0.05, 0.18, "Size and shade both encode the measure.\n"
            "Top 3 nodes are labeled.", fontsize=9.5, color=INK_2,
            transform=ax.transAxes, va="top")

    fig.suptitle("Karate club: who is central depends on how you measure it",
                 x=0.02, ha="left", fontsize=16, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_heatmap(df, path):
    measures = list(LABELS)
    order = (df.assign(_hi=df["club"] == "Mr. Hi")
               .sort_values(["_hi", "degree"], ascending=[False, False]).index)
    data = df.loc[order, measures].apply(_minmax)

    fig, ax = plt.subplots(figsize=(6.4, 11))
    im = ax.imshow(data.values, cmap=CMAP, aspect="auto", vmin=0, vmax=1)
    # 2px-style surface gaps between cells
    ax.set_xticks([x - 0.5 for x in range(1, len(measures))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(order))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(measures)), [LABELS[m] for m in measures], rotation=40, ha="right")
    ax.set_yticks(range(len(order)), [str(n) for n in order], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    n_hi = int((df["club"] == "Mr. Hi").sum())
    ax.axhline(n_hi - 0.5, color=INK, linewidth=1.5)
    ax.text(len(measures) - 0.4, n_hi / 2 - 0.5, "Mr. Hi", rotation=-90,
            va="center", color=INK_2, fontsize=10)
    ax.text(len(measures) - 0.4, n_hi + (len(order) - n_hi) / 2 - 0.5, "Officer",
            rotation=-90, va="center", color=INK_2, fontsize=10)
    ax.set_ylabel("Node (sorted by degree within faction)")

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.09, ticks=[0, 1])
    cb.ax.set_xticklabels(["lowest", "highest"])
    cb.outline.set_visible(False)
    ax.set_title("Centrality by node, scaled per measure", loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    out = Path("figures")
    out.mkdir(exist_ok=True)
    G = nx.karate_club_graph()
    df = compute_karate_centralities(G)
    plot_networks(G, df, out / "karate_centrality_networks.png")
    plot_heatmap(df, out / "karate_centrality_heatmap.png")
    print("Saved figures to", out.resolve())
