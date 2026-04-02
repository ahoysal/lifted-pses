import torch_geometric.datasets as geom_datasets

def load_cora(transform):
    return geom_datasets.Planetoid(root="data/", name="cora", pre_transform=transform, force_reload=True)

def load_lrgb(transform):
    return {
        "train": geom_datasets.LRGBDataset(root="data/", name="Peptides-func", split="train", pre_transform=transform, force_reload=True),
        "test": geom_datasets.LRGBDataset(root="data/", name="Peptides-func", split="test", pre_transform=transform, force_reload=False),
        "val": geom_datasets.LRGBDataset(root="data/", name="Peptides-func", split="val", pre_transform=transform, force_reload=False),
    }