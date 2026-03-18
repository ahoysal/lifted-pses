import torch
import numpy as np

from torch_geometric.nn.models import GCN
print("done torch_geometric")
import datasets
print("done datasets")
# import liftings
# import pses
# import models
import training
import configs

def runExperiement(cfg : configs.Configs):
    # lift and do positional structural encodings
    import torch_geometric.transforms as T
    import torch.nn.functional as F
    def transform(data):
        # lifted = liftings.makeHGFormanRicci(data)
        # T.AddRandomWalkPE(walk_length=cfg.rwpe_len)(data)
        # lifted = liftings.makeHG(data)
        data.x = data.x.float()
        
        # num_anchors = min(cfg.rwpe_anchors, lifted.num_nodes)
        # anchor_nodes = np.random.choice(lifted.num_nodes, num_anchors, replace=False)
        # pse = pses.anchor_positional_encoding(lifted, anchor_nodes, cfg.rwpe_len)
        # if num_anchors < cfg.rwpe_anchors:
        #     padding_size = cfg.rwpe_anchors - num_anchors
        #     pse = F.pad(pse, (0, padding_size, 0, 0), "constant", 0)
        
        # data.x = torch.cat([data.x, pse], dim=1)
        # data.pse = pse
        return data

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.load_lrgb(transform=transform)
    trainDataset = dataset["train"] if isinstance(dataset, dict) else dataset
    # pass dataset to model and train
    # model = models.GPS(
    #      channels=128,
    #      pe_dim=cfg.rwpe_anchors,
    #      num_layers=cfg.layers,
    #      attn_type="performer",
    #      attn_kwargs=None
    # )
    # model = models.GraphNodeTransformer(
    #     in_dim=trainDataset.num_features,
    #     d_model=cfg.embedded,
    #     nhead=cfg.heads,
    #     num_layers=cfg.layers,
    #     out_dim=trainDataset.num_classes,
    #     dropout=cfg.dropout
    # )
    model = GCN(
        in_channels=trainDataset.num_features,
        hidden_channels=cfg.embedded,
        num_layers=cfg.layers,
        out_channels=trainDataset.num_classes,
        dropout=cfg.dropout
    )

    print("Training... (%d parameters)" % (sum(p.numel() for p in model.parameters())))
    training.train(model, dataset, cfg.epochs)

if __name__ == '__main__':
    cfg = configs.Configs()
    cfg.layers = 1
    cfg.rwpe_anchors = 20
    runExperiement(cfg)