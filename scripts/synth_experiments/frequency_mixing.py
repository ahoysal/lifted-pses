"""
Frequency-mixing experiment.

Do graphs where the lifted operator mixes graph-Laplacian frequencies more strongly
also show larger prediction-error improvements from lifted PEs over standard RWPE?

Pipeline
────────
1. Generate a family of N graphs with FIXED degree sequence, VARYING triangle density,
   via degree-preserving edge rewires that progressively enrich or deplete triangles.
2. For each graph compute two mixing scores:
     score(M) = 1 - Σ_i (u_i^T M u_i)^2 / ‖U^T M U‖_F^2
   where u_i are graph-Laplacian eigenvectors and M is:
     • M_Hodge    = B1 (B2 B2^T) B1^T  (upper-Hodge contribution projected to nodes)
     • M_LiftedRW = Forman-Ricci hypergraph random-walk transition matrix
3. Train a GraphNodeTransformer with RWPE and with Hodge PE on an 80/20 split.
   Collect per-graph validation MAE for each model.
4. Δ(g) = MAE_RWPE(g) - MAE_Hodge(g)
5. Compute Pearson / Spearman corr(mixing_score, Δ); save scatter plot and CSV.

Usage
─────
  python frequency_mixing.py [--n N] [--p P] [--num-graphs G] [--trials T]
                              [--out-dir DIR] [--seed S]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import time
import numpy as np
import scipy.stats
import scipy.linalg
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import xgi
from pathlib import Path
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from bruteforce_tasks import count_four_cliques
from models import GraphNodeTransformer
from pses import addRWPE, addHodgePE, hypergraph_random_walk
from liftings import build_hodge_matrices, makeHGFormanRicci
import xgi.convert


# ── Config defaults ───────────────────────────────────────────────────────────

DEFAULT = dict(
    n           = 20,       # nodes per graph
    p           = 0.35,     # edge probability for base ER graph
    num_graphs  = 500,      # total graphs to generate (before flattening)
    flatten_target = None,  # if set, subsample to this many graphs via histogram flattening
    flatten_bins   = 20,    # number of triangle-count bins for flattening
    rewires     = 150,      # degree-preserving rewires between samples
    trials      = 3,        # independent train/eval repetitions per PE type
    epochs      = 80,
    patience    = 15,
    batch_size  = 64,
    lr          = 1e-3,
    d_model     = 64,
    nhead       = 4,
    num_layers  = 3,
    dropout     = 0.1,
    pe_len      = 8,
    rw_anchors  = 16,
    rw_len      = 3,
    train_frac  = 0.8,
    seed        = 42,
)


# ── Graph family generation ───────────────────────────────────────────────────

def _adj_sets(g):
    adj = {v: set(g.neighbors(v)) for v in g.nodes()}
    return adj

def _tri_through_edge(adj, u, v):
    return len(adj[u] & adj[v])

def _rewire_step(g, edges, adj, direction, rng, max_tries=40):
    """In-place degree-preserving rewire.
    direction: +1 = prefer triangle increase, -1 = prefer decrease, 0 = accept any.
    Returns True if a swap was accepted.
    """
    n_edges = len(edges)
    for _ in range(max_tries):
        i, j = rng.choice(n_edges, size=2, replace=False)
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue
        # Try both swap orientations
        for (u, v) in [(a, d), (a, c)]:
            w = d if u == a and v == d else b  # other new edge's first node
            x = c if u == a and v == d else d  # other new edge's second node
            if g.has_edge(u, v) or g.has_edge(w, x):
                continue
            delta = (_tri_through_edge(adj, u, v) + _tri_through_edge(adj, w, x)
                     - _tri_through_edge(adj, a, b) - _tri_through_edge(adj, c, d))
            if direction == 0 or direction * delta >= 0:
                g.remove_edge(a, b); g.remove_edge(c, d)
                g.add_edge(u, v);    g.add_edge(w, x)
                edges[i] = (u, v);   edges[j] = (w, x)
                adj[a].discard(b);   adj[b].discard(a)
                adj[c].discard(d);   adj[d].discard(c)
                adj[u].add(v);       adj[v].add(u)
                adj[w].add(x);       adj[x].add(w)
                return True
    return False

def generate_family(n, p, num_graphs, rewires_per_step, seed, mode='three_way'):
    """Generate a family of graphs with identical degree sequence, varying triangles.

    mode='three_way' (default):
        Splits num_graphs into thirds: triangle-depleting rewires, random rewires,
        triangle-enriching rewires — all starting from the same base graph.

    mode='enrich_only':
        First burns in a large number of depleting rewires to push the base graph
        near its minimum triangle count, then generates all num_graphs samples
        using only triangle-enriching rewires.  This covers the full triangle range
        in a single monotone sweep, avoiding the plateau cluster that forms when the
        depleting run in 'three_way' converges to the minimum.

    Returns list of networkx Graph objects sorted by triangle count.
    """
    rng = np.random.default_rng(seed)
    base = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1 << 31)))
    # Ensure connected; if not, try again a few times
    for _ in range(10):
        if nx.is_connected(base):
            break
        base = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1 << 31)))

    family = []

    if mode == 'three_way':
        n_each = num_graphs // 3
        counts = [n_each, n_each, num_graphs - 2 * n_each]
        directions = [-1, 0, 1]  # deplete, random, enrich
        for direction, count in zip(directions, counts):
            g = base.copy()
            edges = list(g.edges())
            adj = _adj_sets(g)
            for _ in range(count):
                for _ in range(rewires_per_step):
                    _rewire_step(g, edges, adj, direction, rng)
                family.append(g.copy())

    elif mode == 'enrich_only':
        g = base.copy()
        edges = list(g.edges())
        adj = _adj_sets(g)
        # burn-in: deplete toward minimum before collecting any samples
        burn_in = num_graphs * rewires_per_step
        for _ in range(burn_in):
            _rewire_step(g, edges, adj, direction=-1, rng=rng)
        # collect all samples via enriching rewires only
        for _ in range(num_graphs):
            for _ in range(rewires_per_step):
                _rewire_step(g, edges, adj, direction=1, rng=rng)
            family.append(g.copy())

    else:
        raise ValueError(f"Unknown mode {mode!r}. Choose 'three_way' or 'enrich_only'.")

    family.sort(key=lambda g: sum(nx.triangles(g).values()) // 3)
    return family


# ── Mixing score ──────────────────────────────────────────────────────────────

def graph_laplacian_eigvecs(g):
    """Eigenvectors of the combinatorial graph Laplacian, sorted by eigenvalue."""
    L = nx.laplacian_matrix(g).toarray().astype(np.float64)
    _, U = scipy.linalg.eigh(L)   # columns are eigenvectors, ascending eigenvalue
    return U  # (n, n)

def mixing_score(M, U):
    """Frequency-mixing score of operator M relative to Laplacian eigenbasis U.

    score = 1 - Σ_i (u_i^T M u_i)^2 / ‖U^T M U‖_F^2

    High score → M is incoherent with the graph Laplacian (mixes frequencies strongly).
    Score = 0  → M is diagonal in the Laplacian basis (pure graph-spectral operator).
    """
    V = U.T @ M @ U          # representation of M in Laplacian eigenbasis
    frob_sq = np.sum(V ** 2)
    if frob_sq < 1e-12:
        return 0.0
    diag_sq = np.sum(np.diag(V) ** 2)
    return float(1.0 - diag_sq / frob_sq)

def hodge_mixing_score(g):
    """Mixing score for the upper Hodge operator in edge space.

    Computes mixing_score(B2 B2^T, V_e) where V_e are eigenvectors of B1^T B1.

    NOTE: Using node-space M = B1 @ (B2 B2^T) @ B1^T would always give 0 because
    B1 B2 = 0 (boundary of boundary = 0 in any simplicial complex).  We therefore
    stay in the edge domain and ask how much L1_upper mixes the L1_lower eigenmodes.

    Score = 0 when no triangles are present (B2 is empty) or the upper Hodge is
    already diagonal in the L1_lower basis.  Score → 1 when upper Hodge creates
    strong off-diagonal coupling between edge Laplacian modes.
    """
    ei = torch.tensor(list(g.edges()), dtype=torch.long)
    if ei.numel() == 0:
        return 0.0
    ei = torch.cat([ei, ei.flip(1)], dim=0).t().contiguous()
    data = Data(edge_index=ei, num_nodes=g.number_of_nodes())
    built = build_hodge_matrices(data, max_edges=500)
    if not built or built['B2'].shape[1] == 0:
        return 0.0
    lower = built['lower']   # B1^T B1  (m × m)
    upper = built['upper']   # B2 B2^T  (m × m)
    _, V_e = scipy.linalg.eigh(lower)
    return mixing_score(upper, V_e)

def lifted_rw_mixing_score(g, U):
    """Mixing score for the Forman-Ricci hypergraph random-walk operator."""
    ei = torch.tensor(list(g.edges()), dtype=torch.long)
    if ei.numel() == 0:
        return 0.0
    ei = torch.cat([ei, ei.flip(1)], dim=0).t().contiguous()
    data = Data(edge_index=ei, num_nodes=g.number_of_nodes())
    try:
        hg = makeHGFormanRicci(data)
        inc = xgi.convert.to_incidence_matrix(hg, sparse=False).astype(np.float64)
        if inc.shape[1] == 0:
            return 0.0
        T = hypergraph_random_walk(
            __import__('scipy.sparse', fromlist=['coo_array']).coo_array(inc)
        )
        # T is a scipy sparse matrix; convert to dense for small graphs
        M = T.toarray().astype(np.float64)
        # Trim to node space if hypergraph added nodes (xgi may reindex)
        n = g.number_of_nodes()
        M = M[:n, :n]
        return mixing_score(M, U)
    except Exception:
        return 0.0


# ── Dataset ───────────────────────────────────────────────────────────────────

class GraphFamilyDataset(torch.utils.data.Dataset):
    """Wraps the graph family with precomputed labels and a PE transform."""

    def __init__(self, graphs, labels, transform=None):
        self._graphs    = graphs   # list of PyG Data (edge_index, num_nodes)
        self._labels    = labels   # precomputed list of floats
        self._transform = transform

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, idx):
        data = self._graphs[idx].clone()
        data.y = torch.tensor([self._labels[idx]], dtype=torch.float)
        data.x = torch.ones(data.num_nodes, 1)
        if self._transform is not None:
            data = self._transform(data)
        return data

def nx_to_pyg(g):
    ei = torch.tensor(list(g.edges()), dtype=torch.long)
    ei = torch.cat([ei, ei.flip(1)], dim=0).t().contiguous()
    return Data(edge_index=ei, num_nodes=g.number_of_nodes())


# ── Training ──────────────────────────────────────────────────────────────────

def make_transform(pe_type, cfg):
    def t(data):
        if data.x is None:
            data.x = torch.ones(data.num_nodes, 1)
        if pe_type == 'RWPE':
            return addRWPE(data, cfg['rw_anchors'], cfg['rw_len'], data)
        if pe_type == 'Hodge':
            return addHodgePE(data, cfg['pe_len'])
        return data
    return t

def train_and_eval(train_ds, val_ds, cfg, device):
    """Train one model; return per-graph val MAE list (same order as val_ds)."""
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False)

    in_dim = train_ds[0].x.shape[1]
    model = GraphNodeTransformer(
        in_dim=in_dim, d_model=cfg['d_model'], nhead=cfg['nhead'],
        num_layers=cfg['num_layers'], out_dim=1, dropout=cfg['dropout'],
    ).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.L1Loss()

    best_val  = float('inf')
    best_state = None
    patience_ctr = 0

    for _ in range(cfg['epochs']):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = loss_fn(model(batch).squeeze(-1), batch.y.float())
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pred = model(batch).squeeze(-1).float()
                total += (pred - batch.y.float()).abs().sum().item()
                n     += len(batch.y)
        val_mae = total / n

        if val_mae < best_val:
            best_val  = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg['patience']:
                break

    # Reload best and collect per-graph errors
    model.load_state_dict(best_state)
    model.eval()
    per_graph_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(batch).squeeze(-1).float()
            errors = (pred - batch.y.float()).abs().cpu().tolist()
            per_graph_errors.extend(errors)

    return per_graph_errors, best_val


# ── Main ──────────────────────────────────────────────────────────────────────

SMOKE_OVERRIDES = dict(
    n          = 10,
    num_graphs = 12,
    rewires    = 5,
    epochs     = 3,
    patience   = 2,
    trials     = 1,
    batch_size = 4,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',           type=int,   default=DEFAULT['n'])
    parser.add_argument('--p',           type=float, default=DEFAULT['p'])
    parser.add_argument('--num-graphs',  type=int,   default=DEFAULT['num_graphs'])
    parser.add_argument('--rewires',     type=int,   default=DEFAULT['rewires'])
    parser.add_argument('--trials',      type=int,   default=DEFAULT['trials'])
    parser.add_argument('--out-dir',     type=Path,  default=Path('results/frequency_mixing'))
    parser.add_argument('--seed',        type=int,   default=DEFAULT['seed'])
    parser.add_argument('--mode',           type=str,   default='three_way',
                        choices=['three_way', 'enrich_only'],
                        help='Graph family generation mode')
    parser.add_argument('--flatten-target', type=int,   default=None,
                        help='If set, subsample generated graphs to this count via histogram flattening')
    parser.add_argument('--flatten-bins',   type=int,   default=DEFAULT['flatten_bins'],
                        help='Number of triangle-count bins for histogram flattening')
    parser.add_argument('--smoke',       action='store_true',
                        help='Tiny parameters for a quick end-to-end sanity check')
    args = parser.parse_args()

    cfg = {**DEFAULT, 'n': args.n, 'p': args.p, 'num_graphs': args.num_graphs,
           'rewires': args.rewires, 'trials': args.trials, 'seed': args.seed,
           'mode': args.mode, 'flatten_target': args.flatten_target,
           'flatten_bins': args.flatten_bins}
    if args.smoke:
        cfg.update(SMOKE_OVERRIDES)
        print("*** SMOKE TEST — using tiny parameters ***\n")

    run_tag = f"n{cfg['n']}_p{cfg['p']}_g{cfg['num_graphs']}_r{cfg['rewires']}_{cfg['mode']}"
    out_dir = args.out_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:     {device}")
    print(f"n={cfg['n']}, p={cfg['p']}, graphs={cfg['num_graphs']}, "
          f"rewires={cfg['rewires']}, trials={cfg['trials']}")
    print(f"Output:     {out_dir}\n")

    # ── 1. Generate graph family ──────────────────────────────────────────────
    print("Generating graph family...")
    t0 = time.time()
    family = generate_family(cfg['n'], cfg['p'], cfg['num_graphs'], cfg['rewires'], cfg['seed'], mode=cfg['mode'])
    tri_raw = [sum(nx.triangles(g).values()) // 3 for g in family]
    print(f"  Done ({time.time()-t0:.1f}s).  {len(family)} graphs, "
          f"triangle range: {min(tri_raw)} – {max(tri_raw)}")

    if cfg['flatten_target']:
        target     = cfg['flatten_target']
        n_bins     = cfg['flatten_bins']
        max_per_bin = target // n_bins
        tri_arr    = np.array(tri_raw)
        tri_min, tri_max = tri_arr.min(), tri_arr.max()
        rng_flat   = np.random.default_rng(cfg['seed'] + 99)
        bins: dict = {}
        for g, tc in zip(family, tri_arr):
            b = int((tc - tri_min) / (tri_max - tri_min + 1e-9) * n_bins)
            bins.setdefault(b, []).append((g, int(tc)))
        flat = []
        for b in sorted(bins):
            items = bins[b]
            if len(items) > max_per_bin:
                idx   = rng_flat.choice(len(items), max_per_bin, replace=False)
                items = [items[i] for i in idx]
            flat.extend(items)
        flat.sort(key=lambda x: x[1])
        family = [g for g, _ in flat]
        print(f"  Flattened to {len(family)} graphs  "
              f"(target={target}, bins={n_bins}, max_per_bin={max_per_bin})")

    # ── 2. Compute mixing scores and labels ───────────────────────────────────
    print("Computing mixing scores and labels...")
    pyg_graphs, labels, scores_hodge, scores_liftedrw, tri_counts = [], [], [], [], []

    for g in tqdm(family, desc="  graphs"):
        U = graph_laplacian_eigvecs(g)
        scores_hodge.append(hodge_mixing_score(g))
        scores_liftedrw.append(lifted_rw_mixing_score(g, U))

        ei = torch.tensor(list(g.edges()), dtype=torch.long)
        ei = torch.cat([ei, ei.flip(1)], dim=0).t().contiguous()
        pyg_graphs.append(Data(edge_index=ei, num_nodes=g.number_of_nodes()))
        labels.append(float(count_four_cliques(ei, g.number_of_nodes())))
        tri_counts.append(sum(nx.triangles(g).values()) // 3)

    scores_hodge    = np.array(scores_hodge)
    scores_liftedrw = np.array(scores_liftedrw)
    labels          = np.array(labels)
    tri_counts      = np.array(tri_counts)

    print(f"  Hodge mixing:    mean={scores_hodge.mean():.3f}  std={scores_hodge.std():.3f}")
    print(f"  LiftedRW mixing: mean={scores_liftedrw.mean():.3f}  std={scores_liftedrw.std():.3f}")
    print(f"  Label (K4):      mean={labels.mean():.1f}  std={labels.std():.1f}")

    # ── 3. Train/val split ────────────────────────────────────────────────────
    n_train = int(cfg['train_frac'] * len(family))
    rng_split = np.random.default_rng(cfg['seed'] + 1)
    idx = rng_split.permutation(len(family))
    tr_idx, va_idx = idx[:n_train], idx[n_train:]
    split_map = {int(i): 'train' for i in tr_idx}
    split_map.update({int(i): 'val' for i in va_idx})

    # graphs.csv — one row per graph, all graphs, mixing scores + labels
    graphs_csv = out_dir / 'graphs.csv'
    with open(graphs_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'graph_idx', 'split', 'tri_count', 'label_k4',
            'mixing_hodge', 'mixing_liftedrw',
        ])
        w.writeheader()
        for i in range(len(family)):
            w.writerow({
                'graph_idx':      i,
                'split':          split_map[i],
                'tri_count':      int(tri_counts[i]),
                'label_k4':       float(labels[i]),
                'mixing_hodge':   float(scores_hodge[i]),
                'mixing_liftedrw': float(scores_liftedrw[i]),
            })
    print(f"\nSaved: {graphs_csv}")

    # ── 4. Training loop ──────────────────────────────────────────────────────
    # per_graph_errors.csv — one row per (val_graph, trial, pe_type)
    errors_csv = out_dir / 'per_graph_errors.csv'
    trials_csv = out_dir / 'trial_summary.csv'

    with open(errors_csv, 'w', newline='') as ef, \
         open(trials_csv, 'w', newline='') as tf:
        ew = csv.DictWriter(ef, fieldnames=['trial', 'pe_type', 'graph_idx', 'error'])
        tw = csv.DictWriter(tf, fieldnames=['trial', 'pe_type', 'val_mae'])
        ew.writeheader(); tw.writeheader()

        all_delta = np.zeros(len(va_idx))

        for trial in range(cfg['trials']):
            print(f"\nTrial {trial+1}/{cfg['trials']}")
            torch.manual_seed(cfg['seed'] + trial)

            print("  Precomputing Hodge PE...")
            hodge_transform = make_transform('Hodge', cfg)
            train_hodge = GraphFamilyDataset(
                [pyg_graphs[i] for i in tr_idx], labels[tr_idx], hodge_transform)
            val_hodge   = GraphFamilyDataset(
                [pyg_graphs[i] for i in va_idx], labels[va_idx], hodge_transform)
            train_hodge_precomp = [train_hodge[i] for i in range(len(train_hodge))]
            val_hodge_precomp   = [val_hodge[i]   for i in range(len(val_hodge))]
            print("  Done.")

            rwpe_transform = make_transform('RWPE', cfg)
            train_rwpe = GraphFamilyDataset(
                [pyg_graphs[i] for i in tr_idx], labels[tr_idx], rwpe_transform)
            val_rwpe   = GraphFamilyDataset(
                [pyg_graphs[i] for i in va_idx], labels[va_idx], rwpe_transform)

            print("  Training RWPE...")
            errs_rwpe, mae_rwpe = train_and_eval(train_rwpe, val_rwpe, cfg, device)
            print(f"  RWPE  val_MAE={mae_rwpe:.4f}")

            print("  Training Hodge...")
            errs_hodge, mae_hodge = train_and_eval(
                train_hodge_precomp, val_hodge_precomp, cfg, device)
            print(f"  Hodge val_MAE={mae_hodge:.4f}")

            tw.writerow({'trial': trial, 'pe_type': 'RWPE',  'val_mae': f'{mae_rwpe:.6f}'})
            tw.writerow({'trial': trial, 'pe_type': 'Hodge', 'val_mae': f'{mae_hodge:.6f}'})

            for k, gi in enumerate(va_idx):
                ew.writerow({'trial': trial, 'pe_type': 'RWPE',
                             'graph_idx': int(gi), 'error': f'{errs_rwpe[k]:.6f}'})
                ew.writerow({'trial': trial, 'pe_type': 'Hodge',
                             'graph_idx': int(gi), 'error': f'{errs_hodge[k]:.6f}'})

            delta = np.array(errs_rwpe) - np.array(errs_hodge)
            all_delta += delta / cfg['trials']

    print(f"Saved: {errors_csv}")
    print(f"Saved: {trials_csv}")

    # ── 5. Correlation analysis + correlations.csv ────────────────────────────
    print("\n── Correlation results ──")
    corr_rows = []
    for score_name, scores_va in [
        ("mixing_hodge",    scores_hodge[va_idx]),
        ("mixing_liftedrw", scores_liftedrw[va_idx]),
        ("tri_count",       tri_counts[va_idx].astype(float)),
    ]:
        r_p, p_p = scipy.stats.pearsonr( scores_va, all_delta)
        r_s, p_s = scipy.stats.spearmanr(scores_va, all_delta)
        corr_rows.append({
            'predictor':  score_name,
            'pearson_r':  f'{r_p:.4f}', 'pearson_p':  f'{p_p:.4f}',
            'spearman_r': f'{r_s:.4f}', 'spearman_p': f'{p_s:.4f}',
        })
        print(f"  {score_name:<20}  Pearson r={r_p:.3f} (p={p_p:.3f})  "
              f"Spearman ρ={r_s:.3f} (p={p_s:.3f})")

    corr_csv = out_dir / 'correlations.csv'
    with open(corr_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        w.writeheader(); w.writerows(corr_rows)
    print(f"Saved: {corr_csv}")

    # ── 6. Scatter plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [
        (scores_hodge[va_idx],    "Hodge mixing score",    "tab:blue"),
        (scores_liftedrw[va_idx], "LiftedRW mixing score", "tab:orange"),
        (tri_counts[va_idx],      "Triangle count",        "tab:green"),
    ]
    for ax, (x, xlabel, color) in zip(axes, pairs):
        ax.scatter(x, all_delta, alpha=0.4, s=15, color=color)
        r, _ = scipy.stats.pearsonr(x, all_delta)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Δ = MAE_RWPE − MAE_Hodge")
        ax.set_title(f"Pearson r = {r:.3f}")
        ax.axhline(0, color='k', lw=0.8, ls='--')
        m, b = np.polyfit(x, all_delta, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m*xs + b, color=color, lw=1.5)

    plt.suptitle(f"Frequency mixing vs Hodge PE gain  "
                 f"(n={cfg['n']}, p={cfg['p']}, {cfg['num_graphs']} graphs, "
                 f"{cfg['trials']} trials)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / 'scatter.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    print(f"\nAll outputs in: {out_dir}")


if __name__ == '__main__':
    main()
