
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def _to_undirected_if_needed(G):
    # Eigenvector centrality is commonly defined on undirected graphs; 
    # for directed graphs we use the undirected view by default.
    return G.to_undirected() if G.is_directed() else G

def eigenvector_table(G, max_iter=1000, tol=1e-06):
    """
    Return a DataFrame of nodes and their eigenvector centrality, sorted descending.
    """
    Gu = _to_undirected_if_needed(G)
    ec = nx.eigenvector_centrality(Gu, max_iter=max_iter, tol=tol)
    df = pd.DataFrame([{"node": n, "eigenvector": v} for n, v in ec.items()])
    df = df.sort_values("eigenvector", ascending=False).reset_index(drop=True)
    return df

def barplot_top_eigenvector(G, top_k=20, max_iter=1000, tol=1e-06, figsize=(10,6)):
    """
    Bar chart of the top-k nodes by eigenvector centrality.
    """
    df = eigenvector_table(G, max_iter=max_iter, tol=tol).head(top_k)
    plt.figure(figsize=figsize)
    plt.bar(df["node"].astype(str), df["eigenvector"])
    plt.xticks(rotation=60, ha="right")
    plt.title(f"Top {len(df)} Nodes by Eigenvector Centrality")
    plt.xlabel("Node")
    plt.ylabel("Eigenvector centrality")
    plt.tight_layout()
    return df

def draw_graph_with_eigenvector(G, top_k_labels=15, scale=2000, seed=42, max_iter=1000, tol=1e-06, figsize=(10,10)):
    """
    Draw the graph with node sizes proportional to eigenvector centrality and
    labels for the top_k_labels nodes.
    
    Parameters
    ----------
    top_k_labels : int
        Number of highest-centrality nodes to label on the plot (set 0 for no labels).
    scale : float
        Multiplier for node sizes; adjust if nodes look too big/small.
    seed : int
        Seed for layout reproducibility.
    """
    Gu = _to_undirected_if_needed(G)
    ec = nx.eigenvector_centrality(Gu, max_iter=max_iter, tol=tol)
    # Normalize sizes (avoid zeros)
    import numpy as np
    values = np.array(list(ec.values()), dtype=float)
    if values.max() == 0:
        sizes = [300 for _ in values]
    else:
        sizes = (values / values.max()) * scale + 50  # +50 to keep minimum visible
    
    # Layout using spring for consistency
    pos = nx.spring_layout(Gu, seed=seed)
    
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(Gu, pos, node_size=sizes)
    nx.draw_networkx_edges(Gu, pos, alpha=0.3)
    
    if top_k_labels and top_k_labels > 0:
        # Pick top-k nodes to label
        top_nodes = sorted(ec.items(), key=lambda kv: kv[1], reverse=True)[:top_k_labels]
        labels = {n: str(n) for n, _ in top_nodes}
        nx.draw_networkx_labels(Gu, pos, labels=labels, font_size=9)
    
    plt.title("Graph with Node Size ~ Eigenvector Centrality")
    plt.axis("off")
    plt.tight_layout()
    return ec
