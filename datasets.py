import torch_geometric.datasets as geom_datasets

def load_cora(transform, cfg=None):
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = True
    return geom_datasets.Planetoid(root="data/", name="cora", pre_transform=transform, force_reload=True)

def load_lrgb(transform, cfg=None):
    if cfg is not None:
        cfg.multilabel = True
        cfg.classification = True
    return {
        "train": geom_datasets.LRGBDataset(root="data/", name="Peptides-func", split="train", pre_transform=transform, force_reload=True),
        "test": geom_datasets.LRGBDataset(root="data/", name="Peptides-func", split="test", pre_transform=transform, force_reload=False),
        "val": geom_datasets.LRGBDataset(root="data/", name="Peptides-func", split="val", pre_transform=transform, force_reload=False),
    }

def load_csl(transform, cfg=None):
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = True
    return {
        "train": geom_datasets.GNNBenchmarkDataset(root="data/", name="CSL", split="train", pre_transform=transform, force_reload=True),
        "test": geom_datasets.GNNBenchmarkDataset(root="data/", name="CSL", split="test", pre_transform=transform, force_reload=False),
        "val": geom_datasets.GNNBenchmarkDataset(root="data/", name="CSL", split="val", pre_transform=transform, force_reload=False),
    }


def load_zinc(transform, cfg=None):
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = False
    return {
        "train": geom_datasets.ZINC(root="data/ZINC", split="train", subset=True, pre_transform=transform, force_reload=True),
        "test": geom_datasets.ZINC(root="data/ZINC", split="test", subset=True, pre_transform=transform, force_reload=False),
        "val": geom_datasets.ZINC(root="data/ZINC", split="val", subset=True, pre_transform=transform, force_reload=False),
    }