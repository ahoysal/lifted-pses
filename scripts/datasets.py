import torch_geometric.datasets as geom_datasets
import bruteforce

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

    import torch.nn.functional as F

    def zinc_transform(data):
        data.x = F.one_hot(data.x.squeeze(), num_classes=28).float()
        return transform(data) if transform is not None else data

    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = False
    return {
        "train": geom_datasets.ZINC(root="data/ZINC", split="train", subset=True, pre_transform=zinc_transform, force_reload=True),
        "test": geom_datasets.ZINC(root="data/ZINC", split="test", subset=True, pre_transform=zinc_transform, force_reload=False),
        "val": geom_datasets.ZINC(root="data/ZINC", split="val", subset=True, pre_transform=zinc_transform, force_reload=False),
    }

def load_bruteforce(transform, cfg=None):
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = False # Regression on number of triangles
    return {
        "train": bruteforce.BruteforceDataset(root="data/Bruteforce", split="train", pre_transform=transform, force_reload=True),
        "test": bruteforce.BruteforceDataset(root="data/Bruteforce", split="test", pre_transform=transform, force_reload=False),
        "val": bruteforce.BruteforceDataset(root="data/Bruteforce", split="val", pre_transform=transform, force_reload=False),
    }

def load_erdosrenyi(transform, cfg=None):
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = False # Regression on number of 4 cycles
    return {
        "train": bruteforce.ErdosRenyiDataset(root="data/ErdosRenyi", split="train", pre_transform=transform, force_reload=True),
        "test": bruteforce.ErdosRenyiDataset(root="data/ErdosRenyi", split="test", pre_transform=transform, force_reload=False),
        "val": bruteforce.ErdosRenyiDataset(root="data/ErdosRenyi", split="val", pre_transform=transform, force_reload=False),
    }

def load_sds(transform, cfg=None):
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = False # Regression on number of 4 cycles
    return {
        "train": bruteforce.SameDegreeSequenceDataset(root="data/SameDegreeSequence", split="train", pre_transform=transform, force_reload=True),
        "test": bruteforce.SameDegreeSequenceDataset(root="data/SameDegreeSequence", split="test", pre_transform=transform, force_reload=False),
        "val": bruteforce.SameDegreeSequenceDataset(root="data/SameDegreeSequence", split="val", pre_transform=transform, force_reload=False),
    }