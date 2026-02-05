import torch_geometric.datasets as geom_datasets

def load_cora(transform):
    return geom_datasets.Planetoid(root="data/", name="cora", pre_transform=transform, force_reload=True)