class Configs():
    def __init__(self) -> None:
        self.embedded = 128
        self.heads = 2
        self.layers = 2
        self.dropout = 0.2
        self.epochs = 300
        
        # can be of type "RWPE", "LapPE", "RWPELifted"
        self.pseType = ["RWPE"]

        # can be of type "Transformer", "GCN", "MeanGuesser"
        self.modelType = "Transformer"
        self.trials = 5
        self.datasetRoot = None

        self.rwpe_anchors = 20
        self.rwpe_len = 3

        # Edge feature dimension for bond-conditioned attention.
        # 0 = disabled (default, reproduces all existing results exactly).
        # Set to 4 for ZINC (single/double/triple/aromatic bond types).
        self.bond_dim = 0

        # can be of type "ZINC", "SDS", "ErdosRenyi", "Bruteforce"
        self.dataset = "ZINC"
        self.classification = True
        self.multilabel = False

        self.staticModel = False

        self.shuffle = True