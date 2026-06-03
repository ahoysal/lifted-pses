import torch
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import networkx
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, remove_self_loops, coalesce
from torch_geometric.utils import erdos_renyi_graph

def countTriangles(edge_index, num_nodes):
    # 2. Create the adjacency matrix using the true number of nodes
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0

    # 3. Count triangles using the formula: trace(A^3) / 6
    A_cubed = torch.matmul(adj, torch.matmul(adj, adj))
    triangle_count = A_cubed.diag().sum().item() / 6.0

    return int(round(triangle_count))

def countFourCliques(edge_index, num_nodes):
    # 1. Create the adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0
    
    # Ensure no self-loops, which could artificially inflate the triangle counts
    adj.fill_diagonal_(0) 

    # 2. Build a batched 3D tensor B where B[i] is the adjacency matrix 
    # of the subgraph induced by the neighborhood of node i.
    # 
    # Using PyTorch broadcasting:
    # - adj.unsqueeze(0) provides the base adjacency matrix (shape 1xNxN).
    # - adj.unsqueeze(1) and adj.unsqueeze(2) create an outer product mask for 
    #   each node's connections. B[i, j, k] will be 1 if and only if j and k 
    #   are connected AND both are connected to i.
    B = adj.unsqueeze(0) * adj.unsqueeze(1) * adj.unsqueeze(2)

    # 3. Batched matrix multiplication to compute B^3 for all neighborhoods at once
    B_cubed = torch.matmul(B, torch.matmul(B, B))
    
    # 4. Count the triangles in each neighborhood using trace(A^3) / 6
    # B_cubed.diagonal extracts the diagonal for each batched matrix.
    traces = B_cubed.diagonal(dim1=1, dim2=2).sum(dim=1)
    triangles_per_node = traces / 6.0
    
    # 5. A 4-clique contains 4 nodes, meaning it will be counted as 1 triangle 
    # in exactly 4 different neighborhoods. Thus, we divide the total sum by 4.
    four_clique_count = triangles_per_node.sum() / 4.0

    return int(round(four_clique_count.item()))

def countFourCycles(edge_index, num_nodes):
    # 1. Create the adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0

    # 2. Get A^2 (Number of paths of length 2 between i and j)
    A_sq = torch.matmul(adj, adj)

    # 3. We only care about distinct nodes (i != j), so zero out the diagonal
    A_sq.fill_diagonal_(0)

    # 4. For every pair, the number of 4-cycles they form is (A^2_ij choose 2).
    # Since n choose 2 = (n * (n - 1)) / 2, and we double-count opposite pairs, 
    # we divide the total sum by 8.
    four_cycle_count = (A_sq * (A_sq - 1)).sum() / 8.0

    return int(round(four_cycle_count.item()))

class BruteforceDataset(InMemoryDataset):
    splits = ["train", "val", "test"]

    def __init__(self, root:str, split:str = "train", transform=None, pre_transform=None, pre_filter=None, force_reload: bool = False,):
        # 'root' is where the dataset will be saved/loaded from
        assert split in self.splits

        super().__init__(root, transform, pre_transform, pre_filter=pre_filter, force_reload=force_reload)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{split}.pt" for split in self.splits]

    def process(self):
        # This function only runs if 'train.pt' does not exist
        for split in self.splits:
            data_list = []
            
            print("Generating synthetic graphs...")
            num_graphs = 1000 if split == "train" else 200
            for i in range(num_graphs):
                num_nodes = torch.randint(10, 30, (1,)).item()
                
                edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), dtype=torch.long)
                edge_index, _ = remove_self_loops(edge_index)
                edge_index = coalesce(to_undirected(edge_index))

                y = torch.tensor([countTriangles(edge_index, num_nodes)], dtype=torch.float)
                
                data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                data_list.append(data)
                
            # Apply pre-filters and pre-transforms if any are passed
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # Save the dataset to disk
            path = osp.join(self.processed_dir, f'{split}.pt')
            self.save(data_list, path)

class ErdosRenyiDataset(InMemoryDataset):
    splits = ["train", "val", "test"]

    def __init__(self, root:str, split:str = "train", transform=None, pre_transform=None, pre_filter=None, force_reload: bool = False,):
        # 'root' is where the dataset will be saved/loaded from
        assert split in self.splits

        super().__init__(root, transform, pre_transform, pre_filter=pre_filter, force_reload=force_reload)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{split}.pt" for split in self.splits]

    def process(self):
        # This function only runs if 'train.pt' does not exist
        for split in self.splits:
            data_list = []
            
            print("Generating synthetic graphs...")
            num_graphs = 1000 if split == "train" else 200
            for i in range(num_graphs):
                num_nodes = torch.randint(10, 30, (1,)).item()

                edge_prob = torch.empty(1).uniform_(0.1, 0.5).item()
                edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=edge_prob, directed=False)

                y = torch.tensor([countFourCycles(edge_index, num_nodes)], dtype=torch.float)
                
                data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                data_list.append(data)
                
            # Apply pre-filters and pre-transforms if any are passed
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # Save the dataset to disk
            path = osp.join(self.processed_dir, f'{split}.pt')
            self.save(data_list, path)

class SameDegreeSequenceDataset(InMemoryDataset):
    splits = ["train", "val", "test"]

    def __init__(self, root:str, split:str = "train", transform=None, pre_transform=None, pre_filter=None, force_reload: bool = False,):
        # 'root' is where the dataset will be saved/loaded from
        assert split in self.splits

        super().__init__(root, transform, pre_transform, pre_filter=pre_filter, force_reload=force_reload)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{split}.pt" for split in self.splits]

    def process(self):
        SEQUENCE_LENGTH = 10
        # This function only runs if 'train.pt' does not exist
        for split in self.splits:
            data_list = []
            
            print("Generating synthetic graphs...")
            num_graphs = 10000 if split == "train" else 2000
            graphsLeft = num_graphs

            while graphsLeft > 0:
                num_nodes = torch.randint(10, 30, (1,)).item()
                edge_prob = torch.empty(1).uniform_(0.1, 0.5).item()
                edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=edge_prob, directed=False)
                edge_index, _ = remove_self_loops(edge_index)

                data = Data(edge_index=edge_index, num_nodes=num_nodes)
                nxg = to_networkx(data, to_undirected=True)

                for j in range(min(SEQUENCE_LENGTH, graphsLeft)):
                    y = torch.tensor([countFourCliques(edge_index, num_nodes)], dtype=torch.float)
                    data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                    data_list.append(data)
                    graphsLeft -= 1

                    try:
                        nxg = networkx.double_edge_swap(nxg, nswap=1, max_tries=100)
                    except networkx.NetworkXException:
                        break
                
                    edge_index = from_networkx(nxg).edge_index


                
            # Apply pre-filters and pre-transforms if any are passed
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # Save the dataset to disk
            path = osp.join(self.processed_dir, f'{split}.pt')
            self.save(data_list, path)