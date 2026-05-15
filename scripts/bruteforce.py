import torch
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

def countTriangles(edge_index, num_nodes):
    # 2. Create the adjacency matrix using the true number of nodes
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0

    # 3. Count triangles using the formula: trace(A^3) / 6
    A_cubed = torch.matmul(adj, torch.matmul(adj, adj))
    triangle_count = A_cubed.diag().sum().item() / 6.0

    return int(round(triangle_count))

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