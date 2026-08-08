'''
Controlled synthetic experiment: compare positional encodings on 4-cycle and
4-clique prediction using degree-sequence-controlled random graphs (Transformer only).

Iterates trial-outer: all (task × encoding) combos run trial 0, then trial 1, etc.
Within each trial, tasks run in the order defined by TASKS, then encodings in ENCODINGS.
One row per trial is appended to results.csv as it completes. Every 10 epochs (and at
early-stop / final epoch), a row is appended to epoch_losses.csv for loss-curve plotting.

Usage:
  python controlled_synthetic.py [--config path/to/config.yaml] [--resume]
  python controlled_synthetic.py --only-tasks count_four_cliques --only-encodings RWPE Hodge
'''

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from bruteforce import SameDegreeSequenceDataset
from bruteforce_tasks import count_four_cycles, count_four_cliques, count_edges_1_triangle_normalized
from models import GraphNodeTransformer
from pses import addRWPE, addLaplacianPE, addHodgePE, addHodgePERandomized
from liftings import makeHGFormanRicci, makeHGFormanRicciRandomized


TASKS = [
    {"name": "count_four_cycles",              "fn": count_four_cycles},
    {"name": "count_four_cliques",             "fn": count_four_cliques},
    {"name": "count_edges_1_tri_normalized",   "fn": count_edges_1_triangle_normalized},
]

ENCODINGS = ["None", "RWPE", "LapPE", "RWPELifted", "RWPELiftedRandom", "Hodge", "HodgeRandom"]

CSV_FIELDS       = ["task", "encoding", "trial", "val_mae"]
EPOCH_CSV_FIELDS = ["task", "encoding", "trial", "epoch", "train_loss", "val_mae"]


# ── Config ───────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def param_dir_name(ds: dict) -> str:
    r, p = ds['num_nodes_range'], ds['edge_prob_range']
    return (
        f"nTrain{ds['num_train']}_nEval{ds['num_eval']}"
        f"_seqLen{ds['sequence_length']}"
        f"_nodes{r[0]}-{r[1]}"
        f"_prob{p[0]}-{p[1]}"
        f"_nswap{ds['nswap']}"
    )


# ── Transforms ───────────────────────────────────────────────────────────────────

def make_transform(enc_type: str, enc_cfg: dict):
    def transform(data):
        if data.x is None:
            data.x = torch.ones(data.num_nodes, 1)
        if enc_type == "RWPE":
            data = addRWPE(data, enc_cfg['rwpe_anchors'], enc_cfg['rwpe_len'], data)
        elif enc_type == "LapPE":
            data = addLaplacianPE(data, enc_cfg['pe_len'])
        elif enc_type == "RWPELifted":
            hg = makeHGFormanRicci(data)
            data = addRWPE(data, enc_cfg['rwpe_anchors'], enc_cfg['rwpe_len'], hg)
        elif enc_type == "RWPELiftedRandom":
            hg = makeHGFormanRicciRandomized(data)
            data = addRWPE(data, enc_cfg['rwpe_anchors'], enc_cfg['rwpe_len'], hg)
        elif enc_type == "Hodge":
            data = addHodgePE(data, enc_cfg['pe_len'])
        elif enc_type == "HodgeRandom":
            data = addHodgePERandomized(data, enc_cfg['pe_len'])
        return data
    return transform


# ── Dataset ──────────────────────────────────────────────────────────────────────

class RelabeledDataset(torch.utils.data.Dataset):
    """Wraps a cached SDS dataset, relabeling with precomputed labels
    and applying an encoding transform. The underlying graphs are unchanged."""
    def __init__(self, base, labels, transform=None):
        self._base      = base
        self._labels    = labels
        self._transform = transform

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        data = self._base[idx].clone()
        data.y = self._labels[idx]
        if self._transform is not None:
            data = self._transform(data)
        return data


def _precompute_labels(base, task_fn):
    return [
        torch.tensor([task_fn(base[i].edge_index, base[i].num_nodes)], dtype=torch.float)
        for i in range(len(base))
    ]


def load_base_splits(ds_cfg: dict):
    """Load (and cache to disk) the underlying graph splits, shared across tasks and encodings."""
    common = dict(
        root=ds_cfg['root'],
        task_fn=count_four_cliques,
        sequence_length=ds_cfg['sequence_length'],
        num_train=ds_cfg['num_train'],
        num_eval=ds_cfg['num_eval'],
        num_nodes_range=tuple(ds_cfg['num_nodes_range']),
        edge_prob_range=tuple(ds_cfg['edge_prob_range']),
        nswap=ds_cfg['nswap'],
        force_reload=False,
    )
    return SameDegreeSequenceDataset(split='train', **common), SameDegreeSequenceDataset(split='val', **common)


def load_splits(train_base, val_base, task_fn, train_labels, val_labels, transform):
    return (
        RelabeledDataset(train_base, train_labels, transform),
        RelabeledDataset(val_base,   val_labels,   transform),
    )


def input_dim(dataset) -> int:
    return dataset[0].x.shape[1]


# ── Training ─────────────────────────────────────────────────────────────────────

def evaluate_mae(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(batch).squeeze(-1)
            total += (pred.float() - batch.y.float()).abs().sum().item()
            n += batch.y.shape[0]
    return total / n


def train_one_trial(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tr_cfg: dict,
    device: torch.device,
    epoch_bar: tqdm,
    on_epoch_log=None,
) -> float:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr_cfg['lr'])
    loss_fn = nn.L1Loss()
    patience = tr_cfg.get('patience', 20)

    best_mae = float('inf')
    best_state = None
    no_improve = 0

    for epoch_num in range(tr_cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = loss_fn(model(batch).squeeze(-1), batch.y.float())
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        val_mae = evaluate_mae(model, val_loader, device)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        epoch_bar.set_postfix(
            loss=f"{epoch_loss:.4f}", val_mae=f"{val_mae:.4f}",
            best=f"{best_mae:.4f}", pat=f"{no_improve}/{patience}",
        )
        epoch_bar.update(1)

        # log every 10 epochs and always at the stopping epoch
        early_stop = no_improve >= patience
        last_epoch = epoch_num == tr_cfg['epochs'] - 1
        if on_epoch_log and ((epoch_num + 1) % 10 == 0 or early_stop or last_epoch):
            on_epoch_log(epoch_num + 1, epoch_loss, val_mae)

        if early_stop:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_mae


# ── CSV helpers ──────────────────────────────────────────────────────────────────

def load_completed(csv_path: Path) -> set[tuple[str, str, int]]:
    if not csv_path.exists():
        return set()
    completed = set()
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            completed.add((row['task'], row['encoding'], int(row['trial'])))
    return completed


def append_row(csv_path: Path, task: str, encoding: str, trial: int, val_mae: float):
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({"task": task, "encoding": encoding, "trial": trial, "val_mae": f"{val_mae:.6f}"})


def append_epoch_row(
    csv_path: Path, task: str, encoding: str, trial: int,
    epoch: int, train_loss: float, val_mae: float,
):
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "task": task, "encoding": encoding, "trial": trial,
            "epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_mae": f"{val_mae:.6f}",
        })


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent / "config_controlled.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append to existing results.csv and skip already-completed rows. "
             "Without this flag, existing CSV files are overwritten.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Override the results_dir from config (experiment-specific output directory).",
    )
    parser.add_argument(
        "--only-tasks", nargs="+", default=None, metavar="TASK",
        help="Run only these tasks (count_four_cliques, count_four_cycles)",
    )
    parser.add_argument(
        "--only-encodings", nargs="+", default=None, metavar="ENC",
        help="Run only these encodings (None, RWPE, LapPE, RWPELifted, RWPELiftedRandom, Hodge, HodgeRandom)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg, mod_cfg, tr_cfg, enc_cfg = (
        cfg['dataset'], cfg['model'], cfg['training'], cfg['encoding']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = args.out_dir if args.out_dir is not None else Path(cfg['output']['results_dir'])
    out_dir  = base_dir / param_dir_name(ds_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path       = out_dir / "results.csv"
    epoch_csv_path = out_dir / "epoch_losses.csv"

    if args.resume:
        completed = load_completed(csv_path)
        if completed:
            tqdm.write(f"Resuming: {len(completed)} trial(s) already in {csv_path}\n")
    else:
        completed = set()
        # Start fresh — truncate any existing CSV files
        for p in (csv_path, epoch_csv_path):
            if p.exists():
                p.unlink()

    active_tasks     = [t for t in TASKS     if not args.only_tasks     or t['name'] in args.only_tasks]
    active_encodings = [e for e in ENCODINGS if not args.only_encodings or e in args.only_encodings]

    n_trials = tr_cfg['trials']

    print(f"Device:    {device}")
    print(f"Config:    {args.config}")
    print(f"Results:   {csv_path}")
    print(f"Epoch log: {epoch_csv_path}")
    print(f"Tasks:     {[t['name'] for t in active_tasks]}")
    print(f"Encodings: {active_encodings}")
    print(f"Trials:    {n_trials}")
    print(f"Patience:  {tr_cfg.get('patience', 20)}\n")

    PRECOMPUTE_ENCS = {'Hodge', 'HodgeRandom'}

    print("Loading datasets (all task × encoding combinations)...")
    train_base, val_base = load_base_splits(ds_cfg)
    loaders = {}
    for task in active_tasks:
        tqdm.write(f"  Precomputing labels for {task['name']}...")
        train_labels = _precompute_labels(train_base, task['fn'])
        val_labels   = _precompute_labels(val_base,   task['fn'])
        tqdm.write(f"  Done.")
        for enc in active_encodings:
            transform = make_transform(enc, enc_cfg)
            train_ds, val_ds = load_splits(train_base, val_base, task['fn'], train_labels, val_labels, transform)
            if enc in PRECOMPUTE_ENCS:
                tqdm.write(f"  Precomputing {enc} PE for {task['name']}...")
                train_ds = [train_ds[i] for i in range(len(train_ds))]
                val_ds   = [val_ds[i]   for i in range(len(val_ds))]
                tqdm.write(f"  Done.")
            train_loader = DataLoader(train_ds, batch_size=tr_cfg['batch_size'], shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=tr_cfg['batch_size'], shuffle=False)
            loaders[(task['name'], enc)] = (train_ds, train_loader, val_loader)
    print("Datasets ready.\n")

    trial_bar = tqdm(range(n_trials), desc="Trials", position=0, leave=True, unit="trial")
    for trial in trial_bar:
        trial_bar.set_description(f"Trial {trial + 1}/{n_trials}")

        task_bar = tqdm(active_tasks, desc="  Tasks", position=1, leave=False, unit="task")
        for task in task_bar:
            task_name = task['name']
            task_bar.set_description(f"  {task_name}")

            enc_bar = tqdm(active_encodings, desc="    Encodings", position=2, leave=False, unit="enc")
            for enc in enc_bar:
                enc_bar.set_description(f"    {enc}")

                if (task_name, enc, trial) in completed:
                    tqdm.write(f"  trial={trial}  {task_name} / {enc:<20}  [done, skipping]")
                    continue

                train_ds, train_loader, val_loader = loaders[(task_name, enc)]
                in_dim = input_dim(train_ds)

                model = GraphNodeTransformer(
                    in_dim=in_dim, d_model=mod_cfg['embedded'],
                    nhead=mod_cfg['heads'], num_layers=mod_cfg['layers'],
                    out_dim=1, dropout=mod_cfg['dropout'],
                )

                def make_epoch_logger(t, e, tr):
                    def on_epoch_log(epoch, train_loss, val_mae):
                        append_epoch_row(epoch_csv_path, t, e, tr, epoch, train_loss, val_mae)
                    return on_epoch_log

                epoch_bar = tqdm(
                    total=tr_cfg['epochs'], desc="      Epochs",
                    position=3, leave=False, unit="ep",
                )
                val_mae = train_one_trial(
                    model, train_loader, val_loader, tr_cfg, device, epoch_bar,
                    on_epoch_log=make_epoch_logger(task_name, enc, trial),
                )
                epoch_bar.close()

                append_row(csv_path, task_name, enc, trial, val_mae)
                completed.add((task_name, enc, trial))
                tqdm.write(f"  trial={trial}  {task_name} / {enc:<20}  val_mae={val_mae:.4f}")

            enc_bar.close()
        task_bar.close()
    trial_bar.close()
    tqdm.write(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()
