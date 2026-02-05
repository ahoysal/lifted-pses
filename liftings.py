import torch_geometric.data as tg
from torch_geometric.utils.convert import to_networkx
import networkx
import xgi

def find3Cliques(nxg : networkx.Graph):
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


