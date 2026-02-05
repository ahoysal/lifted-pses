import torch
import numpy as np

import datasets
import liftings
import pses
import models
import training
import configs

def runExperiement(cfg : configs.Configs):
    # lift and do positional structural encodings
    def transform(data):
        lifted = liftings.makeHG(data)
        pse = pses.anchor_positional_encoding(lifted, np.random.choice(lifted.num_nodes, cfg.rwpe_anchors, replace=False), cfg.rwpe_len)
        data.x = torch.cat([data.x, pse], dim=1)
        return data

    # Load dataset
    dataset = datasets.load_cora(transform=transform)

    # pass dataset to model and train
    model = models.GraphNodeTransformer(
        in_dim=dataset.num_features,
        d_model=cfg.embedded,
        nhead=cfg.heads,
        num_layers=cfg.layers,
        out_dim=dataset.num_classes,
        dropout=cfg.dropout
    )

    training.train(model, dataset, cfg.epochs)

if __name__ == '__main__':
    cfg = configs.Configs()
    cfg.layers = 1
    runExperiement(cfg)