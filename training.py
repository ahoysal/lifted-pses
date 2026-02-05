import torch
import torch_geometric.data as tg
from torch_geometric.loader import DataLoader

def train(model: torch.nn.Module, dataset: tg.Dataset, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # create a loader for batching
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lossFunction = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x)
            
            loss = lossFunction(out[data.train_mask], data.y[data.train_mask])
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # --- Test/Evaluation Step ---
        model.eval()
        total_correct = 0
        total_test_nodes = 0
        
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data.x)

                pred = out.argmax(dim=1)

                test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                
                total_correct += test_correct.item()
                total_test_nodes += data.test_mask.sum().item()

        test_acc = total_correct / total_test_nodes
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f} | Test Acc: {test_acc:.4f}")