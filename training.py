import torch
import torch_geometric.data as tg
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score
from torch_geometric.nn import global_mean_pool
import numpy as np

def train(model: torch.nn.Module, dataset: tg.Dataset | dict[str, tg.Dataset], epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    masked = not isinstance(dataset, dict)

    # create a loader for batching
    trainDataset = dataset if masked else dataset["train"]
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
    valLoader = DataLoader(trainDataset if masked else dataset["val"], batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # lossFunction = torch.nn.CrossEntropyLoss()
    lossFunction = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in trainLoader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)

            if masked:
                loss = lossFunction(out[data.train_mask], data.y[data.train_mask])
            else:
                # probably graph classification
                loss = lossFunction(out, data.y)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # --- Test/Evaluation Step ---
        model.eval()
        
        if masked:
            total_correct = 0
            total_test_nodes = 0
        else:
            all_preds = []
            all_targets = []
        
        with torch.no_grad():
            for data in valLoader:
                data = data.to(device)

                out = model(data)
                if masked:
                    pred = out.argmax(dim=1)
                    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()

                    total_correct += test_correct.item()
                    total_test_nodes += data.test_mask.sum().item()
                else:
                    all_preds.append(out.detach().cpu().numpy())
                    all_targets.append(data.y.cpu().numpy())

        if masked:
            test_acc = total_correct / total_test_nodes
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f} | Test Acc: {test_acc:.4f}")
        else:
            # Concatenate lists of arrays into single large arrays
            full_preds = np.concatenate(all_preds, axis=0)
            full_targets = np.concatenate(all_targets, axis=0)
            
            # Calculate AP on the entire validation set at once
            test_ap = average_precision_score(full_targets, full_preds)
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f} | Test AP: {test_ap:.4f}")