import experiment
import configs

final = []

def run(model, pse):
    id = "%s with %s" % (model, pse)
    print("Running", id)

    cfg = configs.Configs()
    cfg.modelType = model
    if cfg.modelType == "GCN":
        cfg.layers = 3
        cfg.embedded = 754
        cfg.rwpe_anchors = 20
    cfg.pseType = pse
    cfg.trials = 5

    result = experiment.runExperiement(cfg)
    final.append((id, result))

def printStats(final):
    for i in final:
        print("Summary for", i[0])
        print("\tmean: %f" % i[1].mean())
        print("\tstd: %f" % i[1].std())
        print("\traw:", i[1])

cfg = configs.Configs()


run("Transformer", "None")
run("Transformer", "RWPE")
run("Transformer", "LapPE")
run("Transformer", "RWPELifted")
run("GCN", "None")
run("GCN", "RWPE")
run("GCN", "LapPE")
run("GCN", "RWPELifted")

printStats(final)