import argparse

class Configs():
    def __init__(self) -> None:
        self.embedded = 128
        self.heads = 2
        self.layers = 2
        self.dropout = 0.2
        self.epochs = 100
        
        # can be of type "RWPE", "LapPE", "RWPELifted"
        self.pseType = "RWPE"

        # can be of type "Transformer", "GCN"
        self.modelType = "Transformer"
        self.trials = 5

        self.rwpe_anchors = 20
        self.rwpe_len = 3

def parseMessages():
    parser = argparse.ArgumentParser(description="")

    # 3. Add an optional argument with a flag
    parser.add_argument("--embed", "-e", type=int, default=128,
                        help="Number embdedded dimentions")
    
    parser.add_argument("--heads", "-h", type=int, default=4,
                        help="Number heads")

    parser.add_argument("--layers", "-l", type=int, default=4,
                        help="Number layers")

    # 4. Parse the arguments from the command line
    args = parser.parse_args()

    # 5. Use the parsed arguments in your program
    print(f"{args.message}, {args.name}!")
