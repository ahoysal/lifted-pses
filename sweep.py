import experiment
import configs

LAYERS = [1, 2, 4, 8]
DROPOUT = [0.2]
EPOCHS = [200]

ANCHORS = [0, 5, 10, 100, 1000]
LEN = [1, 3, 5, 9]

cfg = configs.Configs()
for layer in LAYERS:
    cfg.layers = layer
    for dropout in DROPOUT:
        cfg.dropout = dropout
        for epoch in EPOCHS:
            cfg.epochs = epoch
            for anchor in ANCHORS:
                cfg.rwpe_anchors = anchor
                for len in LEN:
                    cfg.rwpe_len = len
                    print("START", layer, dropout, epoch, anchor, len)
                    experiment.runExperiement(cfg)
                    print("END", layer, dropout, epoch, anchor, len)