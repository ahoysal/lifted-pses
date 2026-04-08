"""
Positional encoding utilities for hypergraphs.
Includes Arnoldi encoding (eigenvector-based) and random walk-based encoding.
"""

import numpy as np
import torch
from torch_geometric.data import Data
import xgi
import scipy
from scipy.sparse import coo_array
from torch_geometric.utils import degree

def coo2Sparse(coo) -> torch.Tensor:
    """Convert scipy sparse COO matrix to PyTorch sparse tensor."""
    if not isinstance(coo, scipy.sparse.coo_array):
        coo = coo.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

# One of our additions - implementation needs some tweaking to be optimized for its proposed purpose
def compute_hyperedges_medoids(hypergraph: xgi.Hypergraph, node_features: torch.Tensor, top_k: int = 1):
    """
    Compute medoids for each hyperedge based on node features.
    
    Args:
        hypergraph: xgi Hypergraph object
        node_features: Tensor of node features
        top_k: Number of medoids to return per hyperedge
        
    Returns:
        List of unique medoid node indices
    """
    medoids = []
    
    if isinstance(node_features, torch.Tensor):
        features = node_features.cpu().numpy() if node_features.is_cuda else node_features.numpy()
    else:
        features = node_features
    
    # Create mapping from xgi node IDs to feature indices
    node_list = sorted(list(hypergraph.nodes))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Iterate through hyperedges
    for edge_id in hypergraph.edges:
        edge_nodes = hypergraph.edges.members(edge_id)
        if len(edge_nodes) == 0:
            continue
        
        # Convert edge nodes to indices
        edge_node_indices = [node_to_idx[node] for node in edge_nodes if node in node_to_idx]
        
        if len(edge_node_indices) == 0:
            continue
        
        # Get features for nodes in hyperedge
        edge_features = features[edge_node_indices]
        
        # Compute pairwise distances for nodes in hyperedge
        distances = []
        for i, feat_i in enumerate(edge_features):
            dist_sum = np.sum(np.linalg.norm(edge_features - feat_i, axis=1))
            distances.append((dist_sum, edge_node_indices[i]))
        
        # Sort by distance then node index
        distances.sort(key=lambda x: (x[0], x[1]))
        
        # Get top-k medoids for hyperedge
        for i in range(min(top_k, len(distances))):
            medoid_idx = distances[i][1]
            medoids.append(medoid_idx)
    
    # Remove duplicates
    seen = set()
    unique_medoids = []
    for medoid in medoids:
        if medoid not in seen:
            seen.add(medoid)
            unique_medoids.append(medoid)
    
    return unique_medoids

def hypergraph_random_walk(incidence_matrix):
    """
    Compute random walk transition matrix for hypergraph.
    
    Args:
        incidence_matrix: Hypergraph incidence matrix
        
    Returns:
        Transition matrix for random walks
    """
    C = incidence_matrix.T @ incidence_matrix  # hyper-edge "adjacency" matrix
    # C_hat is defined by just the diagonal of C
    C_hat = C * scipy.sparse.eye_array(C.shape[0])

    adj_mat = incidence_matrix @ incidence_matrix.T
    transition_mat = incidence_matrix @ C_hat @ incidence_matrix.T - adj_mat
    # zero diagonals
    transition_mat.setdiag(0)
    zero_to_one = lambda i: 1 if i == 0 else i
    return transition_mat / np.array([zero_to_one(i) for i in transition_mat.sum(axis=1)])

def graph_transition_matrix(data: Data):
    """
    Compute random walk transition matrix for graph.
    
    Args:
        data: contains .edge_index and .edge_weight
        
    Returns:
        Transition matrix for random walks
    """
    row, col = data.edge_index
    weight = data.edge_weight if hasattr(data, "edge_weight") and data.edge_weight is not None else 1.

    deg = degree(row, data.num_nodes)
    deg = deg.pow(-1)
    deg.masked_fill_(torch.isinf(deg), 0.0)
    edges = deg[row] * weight

    return torch.sparse_coo_tensor(
        data.edge_index, 
        edges,
        size=(data.num_nodes, data.num_nodes)
    )

def anchor_positional_encoding(hypergraph: xgi.Hypergraph | Data, anchor_nodes, iterations):
    """
    Compute anchor-based positional encoding using random walks.
    
    Args:
        hypergraph: xgi Hypergraph object
        anchor_nodes: List of anchor node indices
        iterations: Number of random walk iterations
        
    Returns:
        Positional encoding tensor
    """

    if isinstance(hypergraph, xgi.Hypergraph):
        incidence_matrix = xgi.convert.to_incidence_matrix(hypergraph)
        transition_mat = coo2Sparse(hypergraph_random_walk(incidence_matrix))
    else:
        transition_mat = graph_transition_matrix(hypergraph) # a little funny haha, really should rename this

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transition_mat = transition_mat.to(device)
    
    if len(anchor_nodes) == 0:
        print("No anchor nodes provided, returning zero positional encoding")
        return torch.zeros(size=(hypergraph.num_nodes, 0))
    
    # Initialize anchor node matrix (one-hot encoding of anchor nodes)
    indicies = torch.stack([torch.as_tensor(anchor_nodes), torch.arange(len(anchor_nodes))])
    values = torch.ones(len(anchor_nodes))

    anchor_node = torch.sparse_coo_tensor(
        indicies, 
        values, 
        size=(transition_mat.shape[0], len(anchor_nodes))
    ).to(device)
    
    # Perform random walk for specified iterations
    for _ in range(iterations):
        anchor_node = transition_mat @ anchor_node

    return anchor_node.to_dense().cpu()


def arnoldi_encoding(hypergraph: xgi.Hypergraph, k: int, smallestOnly=True):
    """
    Compute Arnoldi positional encoding using Laplacian eigenvectors.
    
    Args:
        hypergraph: xgi Hypergraph object
        k: Number of eigenvectors to compute
        smallestOnly: If True, only compute smallest eigenvectors. If False, compute both smallest and largest.
        
    Returns:
        Positional encoding tensor with k (or 2k if smallestOnly=False) eigenvectors
    """
    if k == 0: 
        return np.zeros((hypergraph.num_nodes, 0))
    
    laplacian = xgi.linalg.laplacian_matrix.normalized_hypergraph_laplacian(
        hypergraph, 
        sparse=True, 
        index=False
    )

    n = laplacian.shape[0]

    kLargest = torch.from_numpy(np.zeros((hypergraph.num_nodes, 0)))
    if not smallestOnly:
        k = k // 2
        DETERMINISM_EIGS_L = np.random.rand(n, k)
        kLargest = torch.from_numpy(np.real(scipy.sparse.linalg.lobpcg(
            laplacian, DETERMINISM_EIGS_L, largest=True, tol=1e-4
        )[1]))

    DETERMINISM_EIGS_S = np.random.rand(n, k)
    
    kSmallest = torch.from_numpy(np.real(scipy.sparse.linalg.lobpcg(
        laplacian, DETERMINISM_EIGS_S, largest=False, tol=1e-4
    )[1]))

    return torch.cat((kLargest, kSmallest), axis=1).to_sparse()

import torch_geometric.transforms as T
import torch.nn.functional as F
from scipy.sparse.linalg import eigs

def addHodgeLaplacianPE(data, hg : xgi.Hypergraph, PElen):
    totalNodes = data.x.shape[-2]
    trueLen = min(PElen, totalNodes - 1)

    lap = xgi.linalg.laplacian(hg)
    

    lapPE = torch.transpose(lap[:trueLen])

    trueLen = data.lapPE.shape[-1]

    if trueLen < PElen:
        padding_size = PElen - trueLen
        data.lapPE = F.pad(data.lapPE, (0, padding_size, 0, 0), "constant", 0)
    
    data.x = torch.cat([data.x, data.lapPE], dim=1)

    return data

def addLaplacianPE(data, PElen):
    totalNodes = data.x.shape[-2]
    trueLen = min(PElen, totalNodes - 1)

    data = T.AddLaplacianEigenvectorPE(k=trueLen, attr_name="lapPE")(data)
    trueLen = data.lapPE.shape[-1]

    if trueLen < PElen:
        padding_size = PElen - trueLen
        data.lapPE = F.pad(data.lapPE, (0, padding_size, 0, 0), "constant", 0)
    
    data.x = torch.cat([data.x, data.lapPE], dim=1)

    return data

def addRWPE(data, PElen, walkLen):
    totalNodes = data.x.shape[-2]
    trueLen = min(PElen, totalNodes - 1)

    data = T.AddRandomWalkPE(walk_length=walkLen, attr_name="RWPE")(data)
    trueLen = data.RWPE.shape[-1]

    if trueLen < PElen:
        padding_size = PElen - trueLen
        data.RWPE = F.pad(data.RWPE, (0, padding_size, 0, 0), "constant", 0)
    elif trueLen > PElen:
        data.RWPE = data.RWPE[..., :PElen]
    
    data.x = torch.cat([data.x, data.RWPE], dim=1)

    return data
