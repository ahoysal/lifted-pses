import json
import random

import torch
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import networkx
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, remove_self_loops, coalesce
from torch_geometric.utils import erdos_renyi_graph

from bruteforce_tasks import count_triangles, count_four_cliques, count_four_cycles, wiener_index


class _ParamsMixin:
    """Invalidates the processed cache when generation parameters change."""

    def _generation_params(self) -> dict:
        raise NotImplementedError

    def _params_path(self, root: str) -> str:
        return osp.join(root, 'processed', 'params.json')

    def _params_match(self, root: str) -> bool:
        path = self._params_path(root)
        if not osp.exists(path):
            return False
        with open(path) as f:
            return json.load(f) == self._generation_params()

    def _save_params(self):
        with open(self._params_path(self.root), 'w') as f:
            json.dump(self._generation_params(), f, indent=2)


class BruteforceDataset(_ParamsMixin, InMemoryDataset):
    splits = ["train", "val", "test"]

    def __init__(self, root:str, split:str = "train", task_fn=count_triangles, transform=None, pre_transform=None, pre_filter=None, force_reload: bool = False,):
        assert split in self.splits
        self.task_fn = task_fn

        force_reload = force_reload or not self._params_match(root)
        super().__init__(root, transform, pre_transform, pre_filter=pre_filter, force_reload=force_reload)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    def _generation_params(self) -> dict:
        return {"task_fn": self.task_fn.__name__}

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{split}.pt" for split in self.splits]

    def process(self):
        for split in self.splits:
            data_list = []

            print("Generating synthetic graphs...")
            num_graphs = 1000 if split == "train" else 200
            for _ in range(num_graphs):
                num_nodes = torch.randint(10, 30, (1,)).item()

                edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), dtype=torch.long)
                edge_index, _ = remove_self_loops(edge_index)
                edge_index = coalesce(to_undirected(edge_index))

                y = torch.tensor([self.task_fn(edge_index, num_nodes)], dtype=torch.float)

                data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                data_list.append(data)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            path = osp.join(self.processed_dir, f'{split}.pt')
            self.save(data_list, path)

        self._save_params()


class ErdosRenyiDataset(_ParamsMixin, InMemoryDataset):
    splits = ["train", "val", "test"]

    def __init__(self, root:str, split:str = "train", task_fn=count_four_cycles, transform=None, pre_transform=None, pre_filter=None, force_reload: bool = False,):
        assert split in self.splits
        self.task_fn = task_fn

        force_reload = force_reload or not self._params_match(root)
        super().__init__(root, transform, pre_transform, pre_filter=pre_filter, force_reload=force_reload)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    def _generation_params(self) -> dict:
        return {"task_fn": self.task_fn.__name__}

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{split}.pt" for split in self.splits]

    def process(self):
        for split in self.splits:
            data_list = []

            print("Generating synthetic graphs...")
            num_graphs = 1000 if split == "train" else 200
            for _ in range(num_graphs):
                num_nodes = torch.randint(10, 30, (1,)).item()

                edge_prob = torch.empty(1).uniform_(0.1, 0.5).item()
                edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=edge_prob, directed=False)

                y = torch.tensor([self.task_fn(edge_index, num_nodes)], dtype=torch.float)

                data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                data_list.append(data)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            path = osp.join(self.processed_dir, f'{split}.pt')
            self.save(data_list, path)

        self._save_params()


class SameDegreeSequenceDataset(_ParamsMixin, InMemoryDataset):
    splits = ["train", "val", "test"]

    def __init__(self, root:str, split:str = "train", task_fn=count_four_cliques, sequence_length:int = 10, num_train:int = 10000, num_eval:int = 2000, num_nodes_range:tuple = (10, 30), edge_prob_range:tuple = (0.1, 0.5), nswap:int = 1, transform=None, pre_transform=None, pre_filter=None, force_reload: bool = False,):
        assert split in self.splits
        self.task_fn = task_fn
        self.sequence_length = sequence_length
        self.num_train = num_train
        self.num_eval = num_eval
        self.num_nodes_range = num_nodes_range
        self.edge_prob_range = edge_prob_range
        self.nswap = nswap

        force_reload = force_reload or not self._params_match(root)
        super().__init__(root, transform, pre_transform, pre_filter=pre_filter, force_reload=force_reload)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    def _generation_params(self) -> dict:
        return {
            "task_fn": self.task_fn.__name__,
            "sequence_length": self.sequence_length,
            "num_train": self.num_train,
            "num_eval": self.num_eval,
            "num_nodes_range": list(self.num_nodes_range),
            "edge_prob_range": list(self.edge_prob_range),
            "nswap": self.nswap,
        }

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{split}.pt" for split in self.splits]

    def process(self):
        for split in self.splits:
            data_list = []

            print("Generating synthetic graphs...")
            num_graphs = self.num_train if split == "train" else self.num_eval
            graphsLeft = num_graphs

            while graphsLeft > 0:
                num_nodes = torch.randint(self.num_nodes_range[0], self.num_nodes_range[1], (1,)).item()
                edge_prob = torch.empty(1).uniform_(self.edge_prob_range[0], self.edge_prob_range[1]).item()
                edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=edge_prob, directed=False)
                edge_index, _ = remove_self_loops(edge_index)

                data = Data(edge_index=edge_index, num_nodes=num_nodes)
                nxg = to_networkx(data, to_undirected=True)

                for _ in range(min(self.sequence_length, graphsLeft)):
                    y = torch.tensor([self.task_fn(edge_index, num_nodes)], dtype=torch.float)
                    data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                    data_list.append(data)
                    graphsLeft -= 1

                    try:
                        nxg = networkx.double_edge_swap(nxg, nswap=self.nswap, max_tries=100)
                    except networkx.NetworkXException:
                        break

                    edge_index = from_networkx(nxg).edge_index

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            path = osp.join(self.processed_dir, f'{split}.pt')
            self.save(data_list, path)

        self._save_params()


class TreeDataset(_ParamsMixin, InMemoryDataset):
    splits = ["train", "val", "test"]

    def __init__(self, root:str, plus_edges:int=0, split:str = "train", task_fn=wiener_index, transform=None, pre_transform=None, pre_filter=None, force_reload: bool = False,):
        # Note: count_triangles, count_four_cliques, count_four_cycles are always 0
        # on trees (plus_edges=0) since trees contain no cycles — valid but degenerate.
        assert split in self.splits
        self.plus_edges = plus_edges
        self.task_fn = task_fn

        force_reload = force_reload or not self._params_match(root)
        super().__init__(root, transform, pre_transform, pre_filter=pre_filter, force_reload=force_reload)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    def _generation_params(self) -> dict:
        return {
            "task_fn": self.task_fn.__name__,
            "plus_edges": self.plus_edges,
        }

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{split}.pt" for split in self.splits]

    def process(self):
        for split in self.splits:
            data_list = []

            print("Generating trees graphs...")
            num_graphs = 1000 if split == "train" else 200
            for _ in range(num_graphs):
                num_nodes = torch.randint(10, 30, (1,)).item()

                tree : networkx.Graph = networkx.random_unlabeled_tree(n=num_nodes)

                nonEdges = list(networkx.non_edges(graph=tree))
                additionalEdges = random.sample(nonEdges, k=self.plus_edges) if self.plus_edges < len(nonEdges) else nonEdges
                tree.add_edges_from(additionalEdges)

                data = from_networkx(tree)
                data.y = torch.tensor([self.task_fn(data.edge_index, num_nodes)], dtype=torch.float)
                data.num_nodes = num_nodes

                data_list.append(data)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            path = osp.join(self.processed_dir, f'{split}.pt')
            self.save(data_list, path)

        self._save_params()
