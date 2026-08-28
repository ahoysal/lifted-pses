import configs
import argparse
from datasets import DATASETS

final = []

def saveGraph(name, plotReturn, saveTo="results/graphs/default"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    for i in range(plotReturn.shape[1]):
        plt.plot(plotReturn[0, i, :], label=f'Trial {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Validation Metric")
    for i in range(plotReturn.shape[1]):
        plt.plot(plotReturn[1, i, :], label=f'Trial {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Metric')
    plt.legend()

    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"{saveTo}.png")
    plt.close()

def run(dataset, model, pse, datasetRoot=None, bond_dim=0):
    import experiment
    pseString = "+".join(pse)
    id = "%s with %s" % (model, pseString)
    print("Running", id)

    cfg = configs.Configs()
    cfg.dataset = dataset
    cfg.modelType = model
    cfg.pseType = pse
    cfg.trials = 5
    cfg.bond_dim = bond_dim

    cfg.datasetRoot = datasetRoot

    if cfg.modelType == "GCN":
        cfg.layers = 4
        cfg.embedded = 754
        cfg.rwpe_anchors = 20
    if cfg.modelType == "MeanGuesser":
        cfg.trials = 1 # we only need one trial

    result, plotReturn = experiment.runExperiement(cfg)
    # saveGraph(id, plotReturn, saveTo="results/graphs/sds/%s_%s" % (model, pseString))
    
    final.append((id, result))

def printStats(final):
    for i in final:
        print("Summary for", i[0])
        print("\tmean: %f" % i[1].mean())
        print("\tstd: %f" % i[1].std())
        print("\traw:", i[1])

parser = argparse.ArgumentParser(description="Run PSEs on a lifted graph.")
parser.add_argument("-d", "--dataset", type=str, default="ZINC", help=f"Dataset. Can be any in { ', '.join(DATASETS.keys()) }")
parser.add_argument("-m", "--model", type=str, default="MeanGuesser", help="Name of the person")
parser.add_argument("-r", "--dsroot", type=str, default=None, help="Root to store data")
parser.add_argument("--HodgePE", action="store_true", help="Include Hodge Laplacian in PSEs")
parser.add_argument("--RWPE", action="store_true", help="Include RWPEs in PSEs")
parser.add_argument("--LapPE", action="store_true", help="Include LapPE in PSEs")
parser.add_argument("--RWPELifted", action="store_true", help="Include RWPELifted in PSEs")
parser.add_argument("--HodgeLowerPE", action="store_true", help="Include lower Hodge Laplacian PE (B1^T B1 only, no upper term)")
parser.add_argument("--NodeTriCount", action="store_true", help="Include per-node triangle count (log1p-scaled scalar)")
parser.add_argument("--EdgeTriAgg", action="store_true", help="Include per-node edge triangle aggregation (4-d: log1p sum/mean/max + count)")
parser.add_argument("--bond-dim", type=int, default=0,
                    help="Edge-type embedding dimension for bond-conditioned attention. "
                         "0=disabled (default). Set to 4 for ZINC (single/double/triple/aromatic).")

args = parser.parse_args()

pse = []
if args.HodgePE:
    pse.append("Hodge")
if args.RWPE:
    pse.append("RWPE")
if args.LapPE:
    pse.append("LapPE")
if args.RWPELifted:
    pse.append("RWPELifted")
if args.HodgeLowerPE:
    pse.append("HodgeLower")
if args.NodeTriCount:
    pse.append("NodeTriCount")
if args.EdgeTriAgg:
    pse.append("EdgeTriAgg")

print(f"Running dataset {args.dataset} with model {args.model} and pses {pse}, datasetRoot = {args.dsroot}, bond_dim = {args.bond_dim}.")
run(args.dataset, args.model, pse, args.dsroot, bond_dim=args.bond_dim)

printStats(final)