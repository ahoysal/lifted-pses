import time
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
                    lifted = liftings.makeHG(data)
                    data = pses.addRWPE(data, cfg.rwpe_anchors, cfg.rwpe_len, lifted)
                case "Hodge":
                    data = pses.addHodgePE(data, cfg.rwpe_anchors)
                case "HodgeLower":
                    data = pses.addHodgePELower(data, cfg.rwpe_anchors)
                case "NodeTriCount":
                    data = pses.addNodeTriCount(data)
                case "EdgeTriAgg":
                    data = pses.addEdgeTriAgg(data)
                case "None":
                    pass
                case _:
                    print(f"Unknown PSE {i}.")
        
        return data

    # Load dataset (PSE computation happens here via pre_transform)
    print("Loading dataset...")
    if cfg.dataset not in datasets.DATASETS:
        print(f"Failed to find dataset {cfg.dataset}! Returning.")
        return -1, plotReturn
    t0_pse = time.perf_counter()
    dataset = datasets.DATASETS[cfg.dataset](transform=transform, cfg=cfg)
    pse_time = time.perf_counter() - t0_pse

    trainDataset = dataset["train"] if isinstance(dataset, dict) else dataset
    print("Dataset loaded. Num graphs: %d, Num features: %d, Num classes: %d" % (len(trainDataset), trainDataset.num_features, trainDataset.num_classes))
    print(f"TIMING  pse_load={pse_time:.1f}s  (includes PSE pre_transform on train split; 0 if cached)")

    plotReturn = np.empty((2, cfg.trials, cfg.epochs)) # two channels for loss and val.
    out_dim = trainDataset.num_classes if cfg.classification else 1

    trial_train_times = []
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
                    dropout=cfg.dropout,
                    bond_dim=cfg.bond_dim,
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
                print(f"Always predicting {trainDataset.data.y.mean(dim=0)} (stddev {trainDataset.data.y.std(dim=0)})")
                model = models.MeanGuesser(
                    prediction=trainDataset.data.y.mean(dim=0)
                )
                cfg.staticModel = True
            case _:
                print("No model type specified! Exiting.")
                return -1, plotReturn

        print("Training... (%d parameters)" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
        t0_train = time.perf_counter()
        metrics[i], lossGraph, valGraph = training.train(model, dataset, cfg)
        train_time = time.perf_counter() - t0_train
        trial_train_times.append(train_time)
        plotReturn[0, i, :] = lossGraph
        plotReturn[1, i, :] = valGraph
        print(f"TIMING  trial={i}  train={train_time:.1f}s")

        model.cpu()
        # validation.validateOnRS(model, "cpu", transform)

    total_train = sum(trial_train_times)
    total = pse_time + total_train
    print("Summary: mean %f, stddev %f." % (metrics.mean(), metrics.std()))
    print("\t", metrics)
    print(f"TIMING SUMMARY  pse={pse_time:.1f}s  train_all_trials={total_train:.1f}s  total={total:.1f}s")
    return metrics, plotReturn

if __name__ == '__main__':
    cfg = configs.Configs()
    
    # cfg.layers = 3
    # cfg.embedded = 754
    # cfg.rwpe_anchors = 20
    print("params: embedded: %d, heads: %d, layers: %d, dropout: %f, epochs: %d, rwpe_anchors: %d, rwpe_len: %d" % (cfg.embedded, cfg.heads, cfg.layers, cfg.dropout, cfg.epochs, cfg.rwpe_anchors, cfg.rwpe_len))
    
    cfg.pseType = ["Hodge"]
    runExperiement(cfg)