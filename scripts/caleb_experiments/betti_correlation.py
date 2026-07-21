'''
Betti number correlation experiment.

For each combination of task and model+encoding, trains on SameDegreeSequenceDataset,
then analyses how β₁ (independent cycles) of validation graphs correlates with
per-graph model performance.

Regression tasks:     Pearson r between β₁ and per-graph MAE.
Classification tasks: Logistic regression of β₁ → correct/incorrect;
                      reports odds ratio and p-value.

Outputs (per run, keyed by dataset params):
  results/betti_correlation/<param_string>/results.md
  results/betti_correlation/<param_string>/task_<name>.png  (one per task)

Usage:
  python betti_correlation.py [--config path/to/config.yaml]
'''

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pickle
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx

from bruteforce import SameDegreeSequenceDataset
from bruteforce_tasks import (
    count_four_cycles, count_four_cliques, count_n_cycles,
    compute_b1 as compute_b1_fn, is_planar as is_planar_fn,
)
from models import GCN, GraphNodeTransformer, MeanGuesser
from pses import addRWPE, addLaplacianPE, addHodgePE
from liftings import makeHGFormanRicci


TASKS = [
    {"name": "count_four_cycles",  "fn": count_four_cycles,  "type": "regression"},
    {"name": "count_four_cliques", "fn": count_four_cliques, "type": "regression"},
    {"name": "count_n_cycles",     "fn": count_n_cycles,     "type": "regression"},
    {"name": "compute_b1",         "fn": compute_b1_fn,      "type": "regression"},
    {"name": "is_planar",          "fn": is_planar_fn,       "type": "classification"},
]

ENCODINGS   = ["None", "RWPE", "LapPE", "RWPELifted", "Hodge"]
MODEL_TYPES = ["MeanGuesser", "GCN", "Transformer"]


# ── Config ──────────────────────────────────────────────────────────────────────

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


# ── Transforms ──────────────────────────────────────────────────────────────────

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
        elif enc_type == "Hodge":
            data = addHodgePE(data, enc_cfg['pe_len'])
        return data
    return transform


# ── Dataset ─────────────────────────────────────────────────────────────────────

def load_splits(ds_cfg: dict, task_fn, transform):
    common = dict(
        root=ds_cfg['root'],
        task_fn=task_fn,
        sequence_length=ds_cfg['sequence_length'],
        num_train=ds_cfg['num_train'],
        num_eval=ds_cfg['num_eval'],
        num_nodes_range=tuple(ds_cfg['num_nodes_range']),
        edge_prob_range=tuple(ds_cfg['edge_prob_range']),
        nswap=ds_cfg['nswap'],
        transform=transform,
        force_reload=False,
    )
    return (
        SameDegreeSequenceDataset(split='train', **common),
        SameDegreeSequenceDataset(split='val',   **common),
    )


def compute_betti1_array(dataset) -> np.ndarray:
    """β₁ = |E| - |V| + β₀ for each graph, in dataset order."""
    out = []
    for data in tqdm(dataset, desc="  β₁", leave=False, unit="graph"):
        g = to_networkx(data, to_undirected=True)
        b0 = nx.number_connected_components(g)
        out.append(g.number_of_edges() - g.number_of_nodes() + b0)
    return np.array(out)


def training_mean(train_ds) -> float:
    return torch.cat([d.y for d in train_ds]).mean().item()


def input_dim(dataset) -> int:
    return dataset[0].x.shape[1]


# ── Models ───────────────────────────────────────────────────────────────────────

def make_model(model_type: str, in_dim: int, mod_cfg: dict, mean_pred: float) -> nn.Module:
    emb, h, l, drop = mod_cfg['embedded'], mod_cfg['heads'], mod_cfg['layers'], mod_cfg['dropout']
    if model_type == "Transformer":
        return GraphNodeTransformer(in_dim=in_dim, d_model=emb, nhead=h, num_layers=l, out_dim=1, dropout=drop)
    if model_type == "GCN":
        return GCN(in_channels=in_dim, hidden_channels=emb, out_channels=1, num_layers=l)
    if model_type == "MeanGuesser":
        return MeanGuesser(prediction=torch.tensor([mean_pred]))
    raise ValueError(model_type)


# ── Training ─────────────────────────────────────────────────────────────────────

def evaluate_metric(model: nn.Module, loader: DataLoader, is_cls: bool, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).squeeze(-1)
            if is_cls:
                total += ((pred > 0) == batch.y.bool()).sum().item()
            else:
                total += (pred - batch.y.float()).abs().sum().item()
            n += batch.y.shape[0]
    return total / n


def train_one_trial(model: nn.Module, train_loader, val_loader, tr_cfg: dict,
                    is_cls: bool, device, epoch_bar: tqdm) -> tuple[float, nn.Module]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr_cfg['lr'])
    loss_fn = nn.BCEWithLogitsLoss() if is_cls else nn.L1Loss()

    best_metric = 0.0 if is_cls else float('inf')
    best_state  = None

    for _ in range(tr_cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch).squeeze(-1), batch.y.float())
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        val_metric = evaluate_metric(model, val_loader, is_cls, device)
        improved   = val_metric > best_metric if is_cls else val_metric < best_metric
        if improved:
            best_metric = val_metric
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        metric_label = "acc" if is_cls else "MAE"
        epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}", **{f"val_{metric_label}": f"{val_metric:.4f}", f"best_{metric_label}": f"{best_metric:.4f}"})
        epoch_bar.update(1)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_metric, model


@torch.no_grad()
def get_val_outputs(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Per-graph (predictions, labels) from validation loader (must be unshuffled)."""
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).squeeze(-1).cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


# ── Betti correlation ─────────────────────────────────────────────────────────────

def regression_correlation(betti1: np.ndarray, preds: np.ndarray, labels: np.ndarray):
    errors = np.abs(preds - labels)
    if np.std(betti1) < 1e-8 or np.std(errors) < 1e-8:
        return None, None
    r, p = stats.pearsonr(betti1, errors)
    return float(r), float(p)


def classification_odds_ratio(betti1: np.ndarray, preds: np.ndarray, labels: np.ndarray):
    correct = ((preds > 0) == (labels > 0.5)).astype(float)
    if correct.std() < 1e-8:
        return None, None
    try:
        result = sm.Logit(correct, sm.add_constant(betti1.astype(float))).fit(disp=False)
        return float(np.exp(result.params[1])), float(result.pvalues[1])
    except Exception:
        return None, None


# ── Plotting ─────────────────────────────────────────────────────────────────────

def plot_task(task_name: str, is_cls: bool, betti1: np.ndarray,
              model_results: dict, out_path: Path):
    """2-row (GCN / Transformer) × 5-col (encodings) scatter plot."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=False)
    fig.suptitle(f"β₁ vs per-graph performance — {task_name}", fontsize=13)

    for row, model_type in enumerate(["GCN", "Transformer"]):
        for col, enc in enumerate(ENCODINGS):
            ax  = axes[row, col]
            key = f"{model_type}+{enc}"

            if key not in model_results:
                ax.set_visible(False)
                continue

            res        = model_results[key]
            preds_mean = res['preds_mean']
            lab        = res['labels']
            b1_range   = np.linspace(betti1.min(), betti1.max(), 200)

            if is_cls:
                correct = ((preds_mean > 0) == (lab > 0.5)).astype(float)
                ax.scatter(betti1, correct, alpha=0.25, s=6, color='steelblue')
                try:
                    fit = sm.Logit(correct, sm.add_constant(betti1.astype(float))).fit(disp=False)
                    ax.plot(b1_range, fit.predict(sm.add_constant(b1_range)), 'r-', lw=1.5)
                except Exception:
                    pass
                or_v, p_v = res.get('odds_ratio'), res.get('p_value')
                stat = f"OR={or_v:.2f}, p={p_v:.3f}" if or_v is not None else ""
                ax.set_ylabel("Correct (0/1)")
            else:
                errors = np.abs(preds_mean - lab)
                ax.scatter(betti1, errors, alpha=0.25, s=6, color='steelblue')
                m, b_int, *_ = stats.linregress(betti1, errors)
                ax.plot(b1_range, m * b1_range + b_int, 'r-', lw=1.5)
                r_v, p_v = res.get('r'), res.get('p_value')
                stat = f"r={r_v:.2f}, p={p_v:.3f}" if r_v is not None else ""
                ax.set_ylabel("Abs. error")

            ax.set_xlabel("β₁")
            ax.set_title(f"{model_type} + {enc}\n{stat}", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────────────

def write_report(out_path: Path, cfg: dict, all_results: dict):
    ds, tr = cfg['dataset'], cfg['training']
    lines  = [
        "# Betti Number Correlation Experiment\n",
        "## Dataset: SameDegreeSequenceDataset\n",
        "| Parameter | Value |",
        "|---|---|",
        f"| root | `{ds['root']}` |",
        f"| num_train | {ds['num_train']} |",
        f"| num_eval | {ds['num_eval']} |",
        f"| sequence_length | {ds['sequence_length']} |",
        f"| num_nodes_range | {ds['num_nodes_range']} |",
        f"| edge_prob_range | {ds['edge_prob_range']} |",
        f"| nswap | {ds['nswap']} |",
        f"| trials | {tr['trials']} |",
        f"| epochs | {tr['epochs']} |",
        f"| lr | {tr['lr']} |",
        "",
    ]

    for task_name, tdata in all_results.items():
        is_cls        = tdata['is_classification']
        model_results = tdata['model_results']
        metric_name   = "Accuracy" if is_cls else "Val MAE"
        corr_name     = "Odds Ratio (β₁)" if is_cls else "Pearson r (β₁)"

        lines += [f"## Task: `{task_name}`  ({'classification' if is_cls else 'regression'})\n"]

        if 'MeanGuesser' in model_results:
            mg = model_results['MeanGuesser']
            m, s = np.mean(mg['val_metrics']), np.std(mg['val_metrics'])
            lines += [f"**MeanGuesser baseline** — {metric_name}: {m:.4f} ± {s:.4f}\n"]

        for model_type in ["GCN", "Transformer"]:
            lines += [
                f"### {model_type}\n",
                f"| Encoding | {metric_name} (mean ± std) | {corr_name} | p-value |",
                "|---|---|---|---|",
            ]
            for enc in ENCODINGS:
                key = f"{model_type}+{enc}"
                if key not in model_results:
                    continue
                res   = model_results[key]
                m_str = f"{np.mean(res['val_metrics']):.4f} ± {np.std(res['val_metrics']):.4f}"
                if is_cls:
                    or_v  = res.get('odds_ratio')
                    c_str = f"{or_v:.3f}" if or_v is not None else "N/A"
                else:
                    r_v   = res.get('r')
                    c_str = f"{r_v:.3f}" if r_v is not None else "N/A"
                p_v   = res.get('p_value')
                p_str = f"{p_v:.4f}" if p_v is not None else "N/A"
                lines.append(f"| {enc} | {m_str} | {c_str} | {p_str} |")
            lines.append("")

    out_path.write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────────

def save_state(path: Path, all_results: dict):
    with open(path, 'wb') as f:
        pickle.dump(all_results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load state.pkl from output dir and skip already-completed combos",
    )
    parser.add_argument(
        "--only-tasks", nargs="+", default=None, metavar="TASK",
        help="Run only these tasks (by name)",
    )
    parser.add_argument(
        "--only-models", nargs="+", default=None, metavar="MODEL",
        help="Run only these model types (GCN, Transformer)",
    )
    parser.add_argument(
        "--only-encodings", nargs="+", default=None, metavar="ENC",
        help="Run only these encodings (None, RWPE, LapPE, RWPELifted, Hodge)",
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    ds_cfg, mod_cfg, tr_cfg, enc_cfg = (
        cfg['dataset'], cfg['model'], cfg['training'], cfg['encoding']
    )

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir    = Path(cfg['output']['results_dir']) / param_dir_name(ds_cfg)
    state_path = out_dir / "state.pkl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prior state if resuming
    all_results: dict = {}
    if args.resume and state_path.exists():
        with open(state_path, 'rb') as f:
            all_results = pickle.load(f)
        tqdm.write(f"Resumed: {len(all_results)} tasks loaded from {state_path}\n")

    active_tasks     = [t for t in TASKS if not args.only_tasks     or t['name'] in args.only_tasks]
    active_models    = [m for m in ["GCN", "Transformer"] if not args.only_models    or m in args.only_models]
    active_encodings = [e for e in ENCODINGS               if not args.only_encodings or e in args.only_encodings]

    print(f"Device:  {device}")
    print(f"Config:  {args.config}")
    print(f"Results: {out_dir}")
    print(f"Tasks: {len(active_tasks)}  |  Models: {active_models}  |  Encodings: {active_encodings}  |  Trials: {tr_cfg['trials']}\n")

    task_bar = tqdm(active_tasks, desc="Tasks", position=0, leave=True, unit="task")
    for task in task_bar:
        task_name, task_fn, task_type = task['name'], task['fn'], task['type']
        is_cls = task_type == 'classification'
        task_bar.set_description(f"Task: {task_name}")

        # Precompute β₁ for val graphs (structure only — no encoding)
        _, val_raw = load_splits(ds_cfg, task_fn, make_transform("None", enc_cfg))
        betti1_val = compute_betti1_array(val_raw)
        tqdm.write(f"\n[{task_name}] β₁ range: [{betti1_val.min()}, {betti1_val.max()}], mean={betti1_val.mean():.2f}")

        if task_name in all_results:
            model_results = all_results[task_name]['model_results']
            all_results[task_name]['betti1_val'] = betti1_val  # refresh
        else:
            model_results = {}
            all_results[task_name] = {
                "is_classification": is_cls,
                "model_results":     model_results,
                "betti1_val":        betti1_val,
            }

        # Build filtered combo list — MeanGuesser baseline always runs
        combos = ([("MeanGuesser", "None")] +
                  [(mt, enc) for mt in active_models for enc in active_encodings])
        combo_bar = tqdm(combos, desc="  Models", position=1, leave=False, unit="combo")

        for model_type, enc in combo_bar:
            result_key = "MeanGuesser" if model_type == "MeanGuesser" else f"{model_type}+{enc}"
            combo_bar.set_description(f"  {result_key}")

            if result_key in model_results:
                tqdm.write(f"  {result_key:<28}  [skipped — already complete]")
                continue

            transform    = make_transform(enc, enc_cfg)
            train_ds, val_ds = load_splits(ds_cfg, task_fn, transform)
            train_loader = DataLoader(train_ds, batch_size=tr_cfg['batch_size'], shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=tr_cfg['batch_size'], shuffle=False)

            in_dim    = input_dim(train_ds)
            mean_pred = training_mean(train_ds)

            val_metrics: list[float] = []
            trial_preds: list[np.ndarray] = []
            labels_arr: np.ndarray | None = None

            if model_type == "MeanGuesser":
                model  = make_model("MeanGuesser", in_dim, mod_cfg, mean_pred).to(device)
                metric = evaluate_metric(model, val_loader, is_cls, device)
                val_metrics.append(metric)
                model_results[result_key] = {"val_metrics": val_metrics}
                tqdm.write(f"  MeanGuesser  {'acc' if is_cls else 'MAE'}={metric:.4f}")
                write_report(out_dir / "results.md", cfg, all_results)
                save_state(state_path, all_results)
                continue

            trial_bar = tqdm(range(tr_cfg['trials']), desc="    Trials", position=2, leave=False, unit="trial")
            for trial in trial_bar:
                trial_bar.set_description(f"    Trial {trial + 1}/{tr_cfg['trials']}")

                epoch_bar = tqdm(total=tr_cfg['epochs'], desc="      Epochs", position=3, leave=False, unit="ep")
                model     = make_model(model_type, in_dim, mod_cfg, mean_pred)
                metric, trained = train_one_trial(
                    model, train_loader, val_loader, tr_cfg, is_cls, device, epoch_bar
                )
                epoch_bar.close()

                val_metrics.append(metric)
                preds, labels_arr = get_val_outputs(trained, val_loader, device)
                trial_preds.append(preds)

            trial_bar.close()

            preds_mean = np.stack(trial_preds).mean(axis=0)
            mean_m     = float(np.mean(val_metrics))
            std_m      = float(np.std(val_metrics))

            if is_cls:
                or_v, p_v = classification_odds_ratio(betti1_val, preds_mean, labels_arr)
                model_results[result_key] = {
                    "val_metrics": val_metrics, "preds_mean": preds_mean,
                    "labels": labels_arr, "odds_ratio": or_v, "p_value": p_v,
                }
                tqdm.write(f"  {result_key:<28}  acc={mean_m:.4f}±{std_m:.4f}  OR={or_v:.3f}  p={p_v:.4f}" if or_v is not None else f"  {result_key:<28}  acc={mean_m:.4f}±{std_m:.4f}")
            else:
                r_v, p_v = regression_correlation(betti1_val, preds_mean, labels_arr)
                model_results[result_key] = {
                    "val_metrics": val_metrics, "preds_mean": preds_mean,
                    "labels": labels_arr, "r": r_v, "p_value": p_v,
                }
                tqdm.write(f"  {result_key:<28}  MAE={mean_m:.4f}±{std_m:.4f}  r={r_v:.3f}  p={p_v:.4f}" if r_v is not None else f"  {result_key:<28}  MAE={mean_m:.4f}±{std_m:.4f}")
            write_report(out_dir / "results.md", cfg, all_results)
            save_state(state_path, all_results)

        combo_bar.close()

        plot_path = out_dir / f"task_{task_name}.png"
        plot_task(
            task_name, is_cls, betti1_val,
            {k: v for k, v in model_results.items() if k != "MeanGuesser"},
            plot_path,
        )
        tqdm.write(f"  → saved {plot_path.name}")

    task_bar.close()

    tqdm.write(f"\nResults saved to {out_dir / 'results.md'}")
    tqdm.write("Done.")


if __name__ == "__main__":
    main()
