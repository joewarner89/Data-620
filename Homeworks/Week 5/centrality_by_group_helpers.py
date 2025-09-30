
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build_centrality_df(G, attr_name, max_iter=1000, tol=1e-6):
    """Return a DataFrame with node, category attribute, degree centrality,
    and eigenvector centrality for every node in G."""
    deg_cent = nx.degree_centrality(G)
    Gu = G.to_undirected() if G.is_directed() else G
    eig_cent = nx.eigenvector_centrality(Gu, max_iter=max_iter, tol=tol)
    records = []
    for n in G.nodes():
        cat = G.nodes[n].get(attr_name, "Unknown")
        records.append({
            "node": n,
            attr_name: cat,
            "degree_centrality": deg_cent.get(n, np.nan),
            "eigenvector_centrality": eig_cent.get(n, np.nan)
        })
    return pd.DataFrame.from_records(records)

def summary_by_group(df, attr_name):
    """Summary stats per category for both centralities."""
    agg = {
        "degree_centrality": ["count", "mean", "median", "std"],
        "eigenvector_centrality": ["mean", "median", "std"]
    }
    out = df.groupby(attr_name, dropna=False).agg(agg)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.sort_values("degree_centrality_mean", ascending=False)

def boxplot_by_group(df, attr_name, metric="degree_centrality", figsize=(8, 6)):
    """Boxplot of a metric by category."""
    plt.figure(figsize=figsize)
    grouped = df.groupby(attr_name)[metric]
    data = [grouped.get_group(g).dropna().values for g in grouped.groups]
    labels = list(grouped.groups.keys())
    plt.boxplot(data, labels=[str(l) for l in labels], showfliers=False)
    plt.title(f"{metric.replace('_',' ').title()} by {attr_name}")
    plt.xlabel(attr_name.title())
    plt.ylabel(metric.replace('_',' ').title())
    plt.tight_layout()

def bar_means_by_group(df, attr_name, metric="degree_centrality", figsize=(8, 4)):
    """Bar chart of group means for a metric."""
    means = df.groupby(attr_name)[metric].mean().sort_values(ascending=False)
    plt.figure(figsize=figsize)
    plt.bar([str(i) for i in means.index], means.values)
    plt.title(f"Mean {metric.replace('_',' ').title()} by {attr_name}")
    plt.xlabel(attr_name.title())
    plt.ylabel(f"Mean {metric.replace('_',' ').title()}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

def scatter_degree_vs_eigenvector(df, attr_name, annotate_top_k=10, figsize=(7, 6)):
    """Scatter of eigenvector vs degree centrality; optionally annotate top_k by eigenvector."""
    plt.figure(figsize=figsize)
    x = df["degree_centrality"]
    y = df["eigenvector_centrality"]
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel("Degree Centrality")
    plt.ylabel("Eigenvector Centrality")
    plt.title("Eigenvector vs Degree Centrality (all nodes)")
    if annotate_top_k and annotate_top_k > 0:
        top_nodes = df.nlargest(annotate_top_k, "eigenvector_centrality")
        for _, r in top_nodes.iterrows():
            plt.annotate(str(r["node"]), (r["degree_centrality"], r["eigenvector_centrality"]), fontsize=8)
    plt.tight_layout()
