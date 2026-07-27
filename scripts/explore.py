import datasets
import bruteforce
import validation
from torch_geometric.utils import from_networkx

def checkRS():
    g_shrikhande = validation.shrikhande()
    g_rooks = validation.rooks4x4()

    data_shrikhande = from_networkx(g_shrikhande)
    data_rooks = from_networkx(g_rooks)
    
    print(bruteforce.countFourCliques(data_shrikhande.edge_index, data_shrikhande.num_nodes), bruteforce.countFourCliques(data_rooks.edge_index, data_rooks.num_nodes))

def check():
    t0 = datasets.load_tree(None, None, 0)
    print(f"t0 mean, std = {t0['train'].data.y.mean(dim=0)}, {t0['train'].data.y.std(dim=0)}")
    t5 = datasets.load_tree(None, None, 5)
    print(f"t5 mean, std = {t5['train'].data.y.mean(dim=0)}, {t5['train'].data.y.std(dim=0)}")
    t10 = datasets.load_tree(None, None, 10)
    print(f"t10 mean, std = {t10['train'].data.y.mean(dim=0)}, {t10['train'].data.y.std(dim=0)}")
    
if __name__ == '__main__':
    check()