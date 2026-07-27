import torch_geometric.datasets as geom_datasets
import bruteforce

def load_cora(transform, cfg=None):
    root = "data/"
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = True
    return geom_datasets.Planetoid(root="data/", name="cora", pre_transform=transform, force_reload=True)

def load_lrgb(transform, cfg=None):
    root = "data/"
    if cfg is not None:
        cfg.multilabel = True
        cfg.classification = True
    
    return {
        "train": geom_datasets.LRGBDataset(root=root, name="Peptides-func", split="train", pre_transform=transform, force_reload=True),
        "test": geom_datasets.LRGBDataset(root=root, name="Peptides-func", split="test", pre_transform=transform, force_reload=False),
        "val": geom_datasets.LRGBDataset(root=root, name="Peptides-func", split="val", pre_transform=transform, force_reload=False),
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

    root = "data/ZINC"
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = False
        if cfg.datasetRoot is not None:
            root = "data/" + cfg.datasetRoot

    return {
        "train": geom_datasets.ZINC(root=root, split="train", subset=True, pre_transform=zinc_transform, force_reload=True),
        "test": geom_datasets.ZINC(root=root, split="test", subset=True, pre_transform=zinc_transform, force_reload=False),
        "val": geom_datasets.ZINC(root=root, split="val", subset=True, pre_transform=zinc_transform, force_reload=False),
    }

def load_bruteforce(transform, cfg=None):

    root = "data/Bruteforce"
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = False # Regression on number of triangles
        if cfg.datasetRoot is not None:
            root = cfg.datasetRoot
    
    return {
        "train": bruteforce.BruteforceDataset(root="data/Bruteforce", split="train", pre_transform=transform, force_reload=True, cfg=cfg),
        "test": bruteforce.BruteforceDataset(root="data/Bruteforce", split="test", pre_transform=transform, force_reload=False, cfg=cfg),
        "val": bruteforce.BruteforceDataset(root="data/Bruteforce", split="val", pre_transform=transform, force_reload=False, cfg=cfg),
    }

def load_erdosrenyi(transform, cfg=None):
    root = "data/ErdosRenyi"
    if cfg is not None:
        cfg.multilabel = False
        cfg.classification = False # Regression on number of 4 cycles
        if cfg.datasetRoot is not None:
            root = cfg.datasetRoot
    
    return {
        "train": bruteforce.ErdosRenyiDataset(root=root, split="train", pre_transform=transform, force_reload=True, cfg=cfg),
        "test": bruteforce.ErdosRenyiDataset(root=root, split="test", pre_transform=transform, force_reload=False, cfg=cfg),
        "val": bruteforce.ErdosRenyiDataset(root=root, split="val", pre_transform=transform, force_reload=False, cfg=cfg),
    }

def load_sds(transform, cfg=None):
    root = "data/SameDegreeSequence"
    if cfg is not None:
        cfg.multilabel = False
        cfg.shuffle = False
        cfg.classification = False # Regression on number of 4 cycles
        if cfg.datasetRoot is not None:
            root = cfg.datasetRoot
    return {
        "train": bruteforce.SameDegreeSequenceDataset(root=root, split="train", pre_transform=transform, force_reload=True, cfg=cfg),
        "test": bruteforce.SameDegreeSequenceDataset(root=root, split="test", pre_transform=transform, force_reload=False, cfg=cfg),
        "val": bruteforce.SameDegreeSequenceDataset(root=root, split="val", pre_transform=transform, force_reload=False, cfg=cfg),
    }

def load_tree(transform, cfg=None, plus_edges=0):
    root = f"data/Tree+{plus_edges}"
    if cfg is not None:
        cfg.multilabel = False
        cfg.shuffle = False
        cfg.classification = False # Regression on wiener index
        if cfg.datasetRoot is not None:
            root = cfg.datasetRoot
    return {
        "train": bruteforce.TreeDataset(root=root, plus_edges=plus_edges, split="train", pre_transform=transform, force_reload=True, cfg=cfg),
        "test": bruteforce.TreeDataset(root=root, plus_edges=plus_edges, split="test", pre_transform=transform, force_reload=False, cfg=cfg),
        "val": bruteforce.TreeDataset(root=root, plus_edges=plus_edges, split="val", pre_transform=transform, force_reload=False, cfg=cfg),
    }

DATASETS = {
    "ZINC": load_zinc,
    "SDS" : load_sds,
    "ErdosRenyi" : load_erdosrenyi,
    "Bruteforce" : load_bruteforce,
    "CSL" : load_csl,
    "LRGB" : load_lrgb,
    "Tree" : lambda transform, cfg: load_tree(transform=transform, cfg=cfg, plus_edges=0),
    "Cycles5" : lambda transform, cfg: load_tree(transform=transform, cfg=cfg, plus_edges=5),
    "Cycles10" : lambda transform, cfg: load_tree(transform=transform, cfg=cfg, plus_edges=10),
}