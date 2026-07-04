import torch
import numpy as np

from torch_geometric.nn.models import GCN
import datasets
import liftings
import pses
import models
import training
import configs
import validation

def runExperiement(cfg : configs.Configs):
    plotReturn = np.empty((2, cfg.trials, cfg.epochs)) # two channels for loss and val.
    # lift and do positional structural encodings
    def transform(data):
        if not hasattr(data, 'x') or data.x is None:
            data.x = torch.ones((data.num_nodes,1))
        
        data.x = data.x.float()

        for i in cfg.pseType:
            match i:
                case "RWPE":
                    data = pses.addRWPE(data, cfg.rwpe_anchors, cfg.rwpe_len, data)
                case "LapPE":
                    data = pses.addLaplacianPE(data, cfg.rwpe_anchors)
                case "RWPELifted":
                    lifted = liftings.makeHGFormanRicci(data)
                    data = pses.addRWPE(data, cfg.rwpe_anchors, cfg.rwpe_len, lifted)
                case "Hodge":
                    data = pses.addHodgePE(data, cfg.rwpe_anchors)
                case "None":
                    pass
                case _:
                    print(f"Unknown PSE {i}.")
        
        return data

    # Load dataset
    print("Loading dataset...")
    if cfg.dataset not in datasets.DATASETS:
        print(f"Failed to find dataset {cfg.DATASETS}! Returning.")
        return -1, plotReturn
    dataset = datasets.DATASETS[cfg.dataset](transform=transform, cfg=cfg)

    trainDataset = dataset["train"] if isinstance(dataset, dict) else dataset
    print("Dataset loaded. Num graphs: %d, Num features: %d, Num classes: %d" % (len(trainDataset), trainDataset.num_features, trainDataset.num_classes))

    plotReturn = np.empty((2, cfg.trials, cfg.epochs)) # two channels for loss and val.
    out_dim = trainDataset.num_classes if cfg.classification else 1

    metrics = np.empty(cfg.trials)
    for i in range(cfg.trials):
        print("Trial %d:" % i)
        match cfg.modelType:
            case "Transformer":
                model = models.GraphNodeTransformer(
                    in_dim=trainDataset.num_features,
                    d_model=cfg.embedded,
                    nhead=cfg.heads,
                    num_layers=cfg.layers,
                    out_dim=out_dim,
                    dropout=cfg.dropout
                )
                cfg.staticModel = False
            case "GCN":
                model = models.GCN(
                    in_channels=trainDataset.num_features,
                    hidden_channels=cfg.embedded,
                    num_layers=cfg.layers,
                    out_channels=out_dim,
                    dropout=cfg.dropout
                )
                cfg.staticModel = False
            case "MeanGuesser":
                model = models.MeanGuesser(
                    prediction=trainDataset.data.y.mean(dim=0)
                )
                cfg.staticModel = True
            case _:
                print("No model type specified! Exiting.")
                return -1, plotReturn

        print("Training... (%d parameters)" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
        metrics[i], lossGraph, valGraph = training.train(model, dataset, cfg)
        plotReturn[0, i, :] = lossGraph
        plotReturn[1, i, :] = valGraph

        model.cpu()
        # validation.validateOnRS(model, "cpu", transform)
    
    print("Summary: mean %f, stddev %f." % (metrics.mean(), metrics.std()))
    print("\t", metrics)
    return metrics, plotReturn

if __name__ == '__main__':
    cfg = configs.Configs()
    
    # cfg.layers = 3
    # cfg.embedded = 754
    # cfg.rwpe_anchors = 20
    print("params: embedded: %d, heads: %d, layers: %d, dropout: %f, epochs: %d, rwpe_anchors: %d, rwpe_len: %d" % (cfg.embedded, cfg.heads, cfg.layers, cfg.dropout, cfg.epochs, cfg.rwpe_anchors, cfg.rwpe_len))
    
    cfg.pseType = ["Hodge"]
    runExperiement(cfg)