'''
One-time script: parse an existing results.md and write state.pkl so that
betti_correlation.py --resume can pick up where a pre-state.pkl run left off.

Usage:
  python bootstrap_state.py <results_dir>

Example:
  python scripts/synth_experiments/bootstrap_state.py \
    results/betti_correlation/nTrain8000_nEval8000_seqLen5_nodes14-17_prob0.2-0.3_nswap2
'''

import sys
import re
import pickle
import math
from pathlib import Path


def parse_results_md(path: Path) -> dict:
    text  = path.read_text()
    blocks = re.split(r'^## Task: `([^`]+)`', text, flags=re.MULTILINE)
    # blocks: [preamble, task1_name, task1_body, task2_name, task2_body, ...]

    all_results: dict = {}

    for i in range(1, len(blocks), 2):
        task_name = blocks[i].strip().split()[0]  # strip "(regression)" etc.
        body      = blocks[i + 1]
        is_cls    = '(classification)' in blocks[i + 1].split('\n')[0]

        model_results: dict = {}
        all_results[task_name] = {
            'is_classification': is_cls,
            'model_results':     model_results,
            'betti1_val':        None,  # not stored in markdown
        }

        # MeanGuesser baseline
        mg = re.search(r'(?:Val MAE|Accuracy): ([0-9.]+) ± ([0-9.]+)', body)
        if mg:
            mean, std = float(mg.group(1)), float(mg.group(2))
            model_results['MeanGuesser'] = {'val_metrics': _fake_trials(mean, std)}

        # Table rows: | Encoding | value ± std | stat | p-value |
        for row in re.finditer(
            r'^\|\s*(\S+)\s*\|\s*([0-9.]+)\s*±\s*([0-9.]+)\s*\|\s*([0-9.]+|N/A)\s*\|\s*([0-9.]+|N/A)\s*\|',
            body, re.MULTILINE,
        ):
            enc, mean_s, std_s, stat_s, p_s = row.groups()
            mean = float(mean_s)
            std  = float(std_s)
            stat = None if stat_s == 'N/A' else float(stat_s)
            p    = None if p_s    == 'N/A' else float(p_s)

            # Figure out which model section we're in by looking at preceding text
            preceding = body[:row.start()]
            gcn_pos   = preceding.rfind('### GCN')
            tr_pos    = preceding.rfind('### Transformer')
            model     = 'Transformer' if tr_pos > gcn_pos else 'GCN'
            key       = f"{model}+{enc}"

            entry = {'val_metrics': _fake_trials(mean, std)}
            if is_cls:
                entry['odds_ratio'] = stat
                entry['p_value']    = p
            else:
                entry['r']       = stat
                entry['p_value'] = p
            # preds_mean / labels not available from markdown (only needed for plots)
            model_results[key] = entry

    return all_results


def _fake_trials(mean: float, std: float, n: int = 3) -> list:
    """Return an n-element list whose mean==mean and population std==std."""
    if n == 1 or std == 0.0:
        return [mean] * n
    # For n=3: [mean - c, mean, mean + c] where c = std * sqrt(3/2)
    c = std * math.sqrt(n / 2)
    vals = [mean - c, mean, mean + c]
    # pad / trim for arbitrary n
    while len(vals) < n:
        vals.append(mean)
    return vals[:n]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    md_path     = results_dir / 'results.md'
    state_path  = results_dir / 'state.pkl'

    if not md_path.exists():
        print(f"ERROR: {md_path} not found")
        sys.exit(1)

    all_results = parse_results_md(md_path)
    n_tasks  = len(all_results)
    n_combos = sum(len(v['model_results']) for v in all_results.values())
    print(f"Parsed {n_tasks} task(s), {n_combos} combo(s) from {md_path}")

    with open(state_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved state to {state_path}")


if __name__ == '__main__':
    main()
