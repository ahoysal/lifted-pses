import torch
import numpy as np

from torch_geometric.nn.models import GCN
import datasets
import liftings
import pses
import models
import training
import configs

def runExperiement(cfg : configs.Configs):
    # lift and do positional structural encodings
    import torch_geometric.transforms as T
    import torch.nn.functional as F
    def transform(data):
        data.x = data.x.float()

        # return data

        # return pses.addRWPE(data, cfg.rwpe_anchors, cfg.rwpe_len)
        # return pses.addLaplacianPE(data, cfg.rwpe_anchors)
        # lifted = liftings.makeHG(data)
        
        num_anchors = min(cfg.rwpe_anchors, data.num_nodes)
        anchor_nodes = np.random.choice(data.num_nodes, num_anchors, replace=False)
        pse = pses.anchor_positional_encoding(data, anchor_nodes, cfg.rwpe_len)
        if num_anchors < cfg.rwpe_anchors:
            padding_size = cfg.rwpe_anchors - num_anchors
            pse = F.pad(pse, (0, padding_size, 0, 0), "constant", 0)
        
        data.x = torch.cat([data.x, pse], dim=1)

        # data.pse = pse
        return data

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.load_lrgb(transform=transform)
    trainDataset = dataset["train"] if isinstance(dataset, dict) else dataset
    print("Dataset loaded. Num graphs: %d, Num features: %d, Num classes: %d" % (len(trainDataset), trainDataset.num_features, trainDataset.num_classes))

    # pass dataset to model and train
    # model = models.GPS(
    #      channels=128,
    #      pe_dim=cfg.rwpe_anchors,
    #      num_layers=cfg.layers,
    #      attn_type="performer",
    #      attn_kwargs=None
    # )

    print("LapPE Transformer")
    model = models.GraphNodeTransformer(
        in_dim=trainDataset.num_features,
        d_model=cfg.embedded,
        nhead=cfg.heads,
        num_layers=cfg.layers,
        out_dim=trainDataset.num_classes,
        dropout=cfg.dropout
    )

    # print("RWPE GCN")
    # model = models.GCN(
    #     in_channels=trainDataset.num_features,
    #     hidden_channels=cfg.embedded,
    #     num_layers=cfg.layers,
    #     out_channels=trainDataset.num_classes,
    #     dropout=cfg.dropout
    # )

    print("Training... (%d parameters)" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
    training.train(model, dataset, cfg.epochs)

if __name__ == '__main__':
    cfg = configs.Configs()
    
    # cfg.layers = 3
    # cfg.embedded = 754
    # cfg.rwpe_anchors = 20
    print("params: embedded: %d, heads: %d, layers: %d, dropout: %f, epochs: %d, rwpe_anchors: %d, rwpe_len: %d" % (cfg.embedded, cfg.heads, cfg.layers, cfg.dropout, cfg.epochs, cfg.rwpe_anchors, cfg.rwpe_len))
    runExperiement(cfg)