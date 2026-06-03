import networkx

import torch
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import bruteforce

# From wikipedia:
# The Shrikhande graph can be constructed as a Cayley graph. The vertex set is Z_4 x Z_4.
# Two vertices are adjacent if and only if the difference is in 
# {± (1,0), ± (0,1), ± (1,1)}.
def shrikhande():
    tr = networkx.Graph()
    for i in range(4):
        for j in range(4):
            tr.add_node((i, j))
    
    for i in range(4):
        for j in range(4):
            for di, dj in [(1, 0), (0, 1), (1, 1)]:
                tr.add_edge((i, j), ((i + di) % 4, (j + dj) % 4))
                tr.add_edge((i, j), ((i - di) % 4, (j - dj) % 4))
    
    return tr

def rooks4x4():
    tr = networkx.Graph()
    for i in range(4):
        for j in range(4):
            tr.add_node((i, j))
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if k != j:
                    tr.add_edge((i, j), (i, k))
                if k != i:
                    tr.add_edge((i, j), (k, j))
    
    return tr

def validateOnRS(model, device, transform=None):
    g_shrikhande = shrikhande()
    g_rooks = rooks4x4()

    data_shrikhande = from_networkx(g_shrikhande)
    data_rooks = from_networkx(g_rooks)

    data_shrikhande.y = countFourCliques(data_shrikhande.edge_index, data_shrikhande.num_nodes).view(1)
    data_rooks.y = countFourCliques(data_rooks.edge_index, data_rooks.num_nodes).view(1)

    if transform is not None:
        data_shrikhande = transform(data_shrikhande)
        data_rooks = transform(data_rooks)

    graph_list = [data_shrikhande, data_rooks]
    loader = DataLoader(graph_list, batch_size=2, shuffle=False)

    model.eval()

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            print("Validation:" + str(out))