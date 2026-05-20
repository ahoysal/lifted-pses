import torch
import torch_geometric.data as tg
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score
from torch_geometric.nn import global_mean_pool
import numpy as np
import copy

import configs
import validation

def evaluate(loader, model, multilabel, classification, masked, device):
    model.eval()
    
    # for regression
    accumulated_mae = 0
    number_graphs = 0

    # for single label
    total_correct = 0
    total_test_nodes = 0

    # for multilabel
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data)
            if not classification:
                accumulated_mae += (out - data.y.view(-1, 1)).abs().sum().item()
                number_graphs += data.num_graphs
            elif not multilabel:
                pred = out.argmax(dim=1)
                test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum() if masked else (pred == data.y).sum()

                total_correct += test_correct.item()
                total_test_nodes += (data.test_mask.sum().item() if masked else data.y.size(0))
            else:
                all_preds.append(out.detach().cpu().numpy())
                all_targets.append(data.y.cpu().numpy())

    if not classification:
        return accumulated_mae / number_graphs
    elif not multilabel:
        return total_correct / total_test_nodes
    else:
        # Concatenate lists of arrays into single large arrays
        full_preds = np.concatenate(all_preds, axis=0)
        full_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate AP on the entire validation set at once
        return average_precision_score(full_targets, full_preds)

def train(model: torch.nn.Module, dataset: tg.Dataset | dict[str, tg.Dataset], cfg : configs.Configs):
    epochs = cfg.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best = (0, None)

    masked = not isinstance(dataset, dict)
    multilabel = False # TODO: infer/pass this in
    classification = False # TODO: infer/pass this in

    # create a loader for batching
    trainDataset = dataset if masked else dataset["train"]
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
    valLoader = DataLoader(trainDataset if masked else dataset["val"], batch_size=32, shuffle=True)
    testLoader = DataLoader(trainDataset if masked else dataset["test"], batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    lossFunction = None
    if classification:
        if multilabel:
            lossFunction = torch.nn.BCEWithLogitsLoss()
        else:
            lossFunction = torch.nn.CrossEntropyLoss()
    else:
        lossFunction = torch.nn.L1Loss()

    lossGraph = []
    valMetricGraph = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in trainLoader:
            data = data.to(device)
            
            if not cfg.staticModel:
                optimizer.zero_grad()

            out = model(data)

            if masked:
                loss = lossFunction(out[data.train_mask], data.y[data.train_mask])
            else:
                # probably graph classification
                loss = lossFunction(out, data.y.view(-1, 1) if not classification else data.y)
            
            loss.backward()
            if not cfg.staticModel:
                optimizer.step()

            total_loss += loss.item()
        
        # --- Test/Evaluation Step ---
        metric = evaluate(valLoader, model, multilabel, classification, masked, device)

        print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f} | Val {('Acc' if not multilabel else 'AP')}: {metric:.4f}")

        lossGraph.append(total_loss)
        valMetricGraph.append(metric)

        if best[1] is None or (metric > best[0] and classification) or (metric < best[0] and not classification):
            best = (metric, copy.deepcopy(model.state_dict()))
            print("[TEST]: better %f!" % metric)

    if best[1] is not None:
        model.load_state_dict(best[1])
    
    metric = evaluate(testLoader, model, multilabel, classification, masked, device)

    print(f"Final Test {('Acc' if not multilabel else 'AP')}: {metric:.4f}")
    validation.validateOnRS(model, device)
    
    return metric, lossGraph, valMetricGraph