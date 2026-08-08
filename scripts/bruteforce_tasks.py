import torch
import networkx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def count_triangles(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0
    A_cubed = torch.matmul(adj, torch.matmul(adj, adj))
    return int(round(A_cubed.diag().sum().item() / 6.0))


def count_four_cliques(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj.fill_diagonal_(0)
    B = adj.unsqueeze(0) * adj.unsqueeze(1) * adj.unsqueeze(2)
    B_cubed = torch.matmul(B, torch.matmul(B, B))
    traces = B_cubed.diagonal(dim1=1, dim2=2).sum(dim=1)
    return int(round((traces / 6.0).sum().item() / 4.0))


def count_four_cycles(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0
    A_sq = torch.matmul(adj, adj)
    A_sq.fill_diagonal_(0)
    return int(round((A_sq * (A_sq - 1)).sum().item() / 8.0))


def count_edges_in_exactly_k_triangles(edge_index, num_nodes, k=1):
    """Count edges that participate in exactly k triangles.
    Directly readable from B2: edge e is in k triangles iff row e of B2 has k non-zeros.
    HodgeRandom scrambles this by assigning random edges to each triangle slot,
    so it cannot recover this quantity even though it preserves the triangle count.
    """
    adj = [set() for _ in range(num_nodes)]
    edges = set()
    for u, v in edge_index.t().tolist():
        if u != v:
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
            adj[u].add(v)
            adj[v].add(u)
    return sum(1 for (u, v) in edges if len(adj[u] & adj[v]) == k)


def make_count_edges_k_triangles(k):
    """Return a task function (edge_index, num_nodes) -> int for a specific k."""
    def fn(edge_index, num_nodes):
        return count_edges_in_exactly_k_triangles(edge_index, num_nodes, k)
    fn.__name__ = f'count_edges_{k}_triangles'
    return fn


count_edges_1_triangle  = make_count_edges_k_triangles(1)
count_edges_2_triangles = make_count_edges_k_triangles(2)


def count_edges_1_triangle_normalized(edge_index, num_nodes):
    """Fraction of triangle-participating edges that are in exactly 1 triangle.
    Independent of total triangle count by construction — HodgeRandom's only leaked
    signal (triangle count) is divided out, leaving only the overlap structure."""
    adj = [set() for _ in range(num_nodes)]
    edges = set()
    for u, v in edge_index.t().tolist():
        if u != v:
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
            adj[u].add(v); adj[v].add(u)
    tri_counts = {e: len(adj[e[0]] & adj[e[1]]) for e in edges}
    edges_in_any = sum(1 for c in tri_counts.values() if c >= 1)
    if edges_in_any == 0:
        return 0.0
    edges_in_exactly_1 = sum(1 for c in tri_counts.values() if c == 1)
    return edges_in_exactly_1 / edges_in_any


def wiener_index(edge_index, num_nodes):
    # Incompatible with datasets that may produce disconnected graphs
    # (BruteforceDataset, ErdosRenyiDataset, SameDegreeSequenceDataset):
    # networkx.wiener_index returns inf for disconnected graphs.
    # TreeDataset always produces connected graphs, so it is always compatible.
    g = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    return networkx.wiener_index(g)

def count_n_cycles(edge_index, num_nodes):
    g = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    # Each undirected cycle appears as 2 directed cycles; simple_cycles on the
    # directed version counts both orientations, so we divide by 2.
    return sum(1 for _ in networkx.simple_cycles(g.to_directed())) // 2

def compute_b1(edge_index, num_nodes):
    g = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    b0 = networkx.number_connected_components(g)
    return g.number_of_edges() - g.number_of_nodes() + b0

def is_planar(edge_index, num_nodes):
    g = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    return 1.0 if networkx.is_planar(g) else 0.0