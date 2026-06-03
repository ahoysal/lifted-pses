import torch_geometric.data as tg
from torch_geometric.utils.convert import to_networkx
import torch
import networkx as nx
import xgi
import numpy as np

def find3Cliques(nxg : nx.Graph):
    triangles = []
    for edge in nxg.edges():
        u, v = edge
        # Common neighbors = triangle
        common = set(nxg.neighbors(u)) & set(nxg.neighbors(v))
        for w in common:
            # Sort to keep unique sets
            triangles.append(tuple(sorted((u, v, w))))
    
    unique_3cliques = list(set(triangles))
    return unique_3cliques

def makeHG(graph : tg.data.BaseData):
    hg = xgi.Hypergraph()
    nxg = to_networkx(graph, to_undirected=True)
    cliques = find3Cliques(nxg)
    hg.add_nodes_from(nxg.nodes())
    hg.add_edges_from(nxg.edges())

    return hg

def unique_undirected_edges(edge_index: torch.Tensor, n: int) -> list[tuple[int, int]]:
    edges = set()
    for u, v in edge_index.cpu().t().tolist():
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if 0 <= a < n and 0 <= b < n:
            edges.add((a, b))
    return sorted(edges)

def undirected_adjacency_sets(edge_index: torch.Tensor, n: int) -> list[set]:
    adj = [set() for _ in range(n)]
    for u, v in unique_undirected_edges(edge_index, n):
        adj[u].add(v)
        adj[v].add(u)
    return adj

def find_triangles(adj: list[set]) -> list[tuple[int, int, int]]:
    """All 3-cliques i<j<k."""
    triangles = []
    n = len(adj)
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            common = adj[i].intersection(adj[j])
            for k in common:
                if k > j:
                    triangles.append((i, j, k))
    return triangles

def build_hodge_matrices(data: tg.Data, max_edges: int) -> dict[str, object]:
    """Build B1, B2, L1 and metadata for clique complex up to dimension 2."""
    n = int(data.num_nodes)
    edges = unique_undirected_edges(data.edge_index, n)
    m = len(edges)
    if n == 0 or m == 0 or m > max_edges:
        return {}

    edge_to_idx = {e: i for i, e in enumerate(edges)}
    adj = undirected_adjacency_sets(data.edge_index, n)
    triangles = find_triangles(adj)

    B1 = np.zeros((n, m), dtype=np.float64)
    for idx, (u, v) in enumerate(edges):
        # fixed orientation u -> v because u < v
        B1[u, idx] = -1.0
        B1[v, idx] = 1.0

    t = len(triangles)
    B2 = np.zeros((m, t), dtype=np.float64)
    for col, (i, j, k) in enumerate(triangles):
        # oriented triangle [i,j,k], boundary [j,k] - [i,k] + [i,j]
        # our edge orientation is always low -> high, so these signs are valid for i<j<k
        B2[edge_to_idx[(j, k)], col] = 1.0
        B2[edge_to_idx[(i, k)], col] = -1.0
        B2[edge_to_idx[(i, j)], col] = 1.0

    lower = B1.T @ B1
    upper = B2 @ B2.T if t > 0 else np.zeros((m, m), dtype=np.float64)
    L1 = lower + upper
    return {
        "n": n,
        "m": m,
        "edges": edges,
        "edge_to_idx": edge_to_idx,
        "triangles": triangles,
        "B1": B1,
        "B2": B2,
        "L1": L1,
        "lower": lower,
        "upper": upper,
    }

def compute_forman_ricci_curvature(G: nx.Graph):
    """
    Computes Forman-Ricci Curvature for unweighted graphs.
    Formula: Ric_F(e) = 4 - deg(u) - deg(v)
    Ref: "A Remedy for Over-Squashing..." Equation (2) simplification.
    """
    curvature = {}
    degrees = dict(G.degree())
    
    for u, v in G.edges():
        # For unweighted graphs, Ric_F is determined by node degrees
        ric_f = 4 - degrees[u] - degrees[v]
        curvature[(u, v)] = ric_f
        
    return curvature

def get_components_by_curvature(G: nx.Graph, curvatures: dict, theta: float, geq: bool = True):
    """
    Extracts connected components (hyperedges) formed by edges satisfying the curvature threshold.
    Ref: Figure 3 and Section 'The remedy'[cite: 106, 123].
    """
    # Create a subgraph with only edges meeting the criteria
    # geq=True -> Ric_F >= theta (Clusters/Red in Fig 3)
    # geq=False -> Ric_F < theta (Backbones/Blue in Fig 3)
    edges_to_keep = [
        edge for edge, curve_val in curvatures.items() 
        if (curve_val >= theta if geq else curve_val < theta)
    ]
    
    subgraph = G.edge_subgraph(edges_to_keep)
    
    # Filter out components with size < 2 (isolated nodes in the subgraph don't form hyperedges)
    components = [list(c) for c in nx.connected_components(subgraph) if len(c) > 1]
    return components

def makeHGFormanRicci(graph: tg.data.BaseData, quantile_p: float = 0.5):
    """
    Lifts a Graph to a Hypergraph using Forman-Ricci Curvature.
    
    Args:
        graph: PyG data object.
        quantile_p: The quantile probability to define threshold theta (default 0.5).
                    Ref: "if p=0.9 then theta would correspond to the... 0.9-th quantile"[cite: 122].
    """
    hg = xgi.Hypergraph()
    nxg = to_networkx(graph, to_undirected=True)
    
    # 1. Add basic nodes
    hg.add_nodes_from(nxg.nodes())
    
    # 2. Compute Curvature for all edges [cite: 108]
    curvatures = compute_forman_ricci_curvature(nxg)
    
    if not curvatures:
        # Fallback if no edges exist
        hg.add_edges_from(nxg.edges())
        return hg
        
    # 3. Determine Threshold Theta
    # "Q_{P_{E}}(p) = theta" [cite: 121]
    curvature_values = list(curvatures.values())
    theta = np.quantile(curvature_values, quantile_p)
    
    # 4. Form Hyperedges via Lifting
    # The paper lifts edges into hyperedges based on curvature distribution.
    # We extract two sets of hyperedges as shown in Figure 3[cite: 106]:
    # Set A: Clusters (Ric >= Theta)
    # Set B: Backbones/Bottlenecks (Ric < Theta)
    
    clusters = get_components_by_curvature(nxg, curvatures, theta, geq=True)
    backbones = get_components_by_curvature(nxg, curvatures, theta, geq=False)
    
    # Add the discovered high-order structures
    # Note: Standard edges are size-2 hyperedges. The paper implies replacing G with H, 
    # but strictly often we keep 1-skeleton (original edges) or just the lifted set. 
    # Here we add the lifted components.
    if clusters:
        hg.add_edges_from(clusters)
    if backbones:
        hg.add_edges_from(backbones)
        
    # Optional: Ensure original pairwise edges are present if the lifting 
    # is meant to augment rather than replace (common in GNN lifting).
    # If strictly following "Graph-to-Hypergraph" replacement, you might comment this out,
    # but for "transform(data)" pipelines, preserving connectivity is usually safer.
    hg.add_edges_from(nxg.edges())

    return hg