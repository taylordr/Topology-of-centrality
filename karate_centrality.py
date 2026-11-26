"""Centrality measures for Zachary's karate club network using NetworkX."""

import networkx as nx
import pandas as pd


def compute_karate_centralities(G=None):
    """Compute common centrality measures for the karate club network.

    Parameters
    ----------
    G : networkx.Graph, optional
        Graph to analyze. Defaults to ``nx.karate_club_graph()``.

    Returns
    -------
    pandas.DataFrame
        One row per node, one column per centrality measure, plus the
        node's ``club`` faction label when available.
    """
    if G is None:
        G = nx.karate_club_graph()

    measures = {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "closeness": nx.closeness_centrality(G),
        "eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
        "pagerank": nx.pagerank(G),
        "katz": nx.katz_centrality_numpy(G, alpha=0.1),
        "harmonic": nx.harmonic_centrality(G),
    }

    df = pd.DataFrame(measures)
    df.index.name = "node"
    clubs = nx.get_node_attributes(G, "club")
    if clubs:
        df.insert(0, "club", pd.Series(clubs))
    return df


if __name__ == "__main__":
    df = compute_karate_centralities()
    pd.set_option("display.width", 120)
    print(df.round(4).to_string())
    print("\nTop 5 nodes by each measure:")
    for col in df.columns.drop("club", errors="ignore"):
        print(f"  {col:12s} {df[col].nlargest(5).index.tolist()}")
