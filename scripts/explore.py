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
    checkRS()
    
if __name__ == '__main__':
    check()