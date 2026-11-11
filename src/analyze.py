"""Baseline analysis: overall metrics (mean +/- std over seeds), per-group results,
popularity bias and catalog coverage. Works for any processed dataset.

python -m src.analyze --dataset beauty                 # test split, results/beauty
python -m src.analyze --data data/processed --out results --split validation   # MovieLens

Groups
  history: user's training-history length, split at its tertiles (short / medium / long)
  target:  held-out item in the head (top 20% of items by training count) or the tail
Popularity bias: share of top-10 recommendations drawn from head items.
Coverage@10: fraction of the catalog appearing in at least one user's top-10.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_data

MODELS = ('popularity', 'deepfm', 'sasrec', 'ngcf')


def hit_rows(top, truth, k):
    matches = top[:, :k] == truth[:, None]
    hits = matches.any(axis=1)
    ranks = np.argmax(matches, axis=1) + 1
    return hits.astype(float), np.where(hits, 1.0 / np.log2(ranks + 1), 0.0)


def load_runs(out: Path, split: str):
    key, truth_key = ('top20', 'validation_truth') if split == 'validation' else ('test_top20', 'test_truth')
    for p in sorted(out.glob('*_top20.npz')):
        m = re.fullmatch(r'([a-z]+)(?:_seed(\d+))?_top20\.npz', p.name)
        if not m or m.group(1) not in MODELS:
            continue
        z = np.load(p)
        if key not in z:
            print(f'skip {p.name}: no {split} rankings (retrain to add them)')
            continue
        yield m.group(1), int(m.group(2) or -1), z[key], z[truth_key]


def main(data, out, split):
    data, out = Path(data), Path(out)
    train, histories, _, _, n_users, n_items = read_data(data)
    pop = np.bincount(train.item_idx.to_numpy(), minlength=n_items)
    head = np.zeros(n_items, dtype=bool)
    head[np.argsort(-pop, kind='stable')[:int(np.ceil(0.2 * n_items))]] = True
    length = np.array([len(h) for h in histories])
    q1, q2 = np.quantile(length, [1 / 3, 2 / 3])
    hist_group = np.full(n_users, '', dtype=object)
    for name, mask in (('short', length <= q1), ('medium', (length > q1) & (length <= q2)), ('long', length > q2)):
        if mask.any():
            hist_group[mask] = f'{name} ({length[mask].min()}-{length[mask].max()})'
    hist_group = hist_group.astype(str)

    overall, groups = [], []
    for model, seed, top, truth in load_runs(out, split):
        tgt_group = np.where(head[truth], 'head', 'tail')
        h10, n10 = hit_rows(top, truth, 10)
        h20, n20 = hit_rows(top, truth, 20)
        top10 = top[:, :10]
        overall.append({'model': model, 'seed': seed, 'Recall@10': h10.mean(), 'NDCG@10': n10.mean(),
                        'Recall@20': h20.mean(), 'NDCG@20': n20.mean(),
                        'HeadShare@10': head[top10].mean(),
                        'Coverage@10': len(np.unique(top10)) / n_items})
        for kind, g in (('history', hist_group), ('target', tgt_group)):
            for value in np.unique(g):
                mask = g == value
                groups.append({'model': model, 'seed': seed, 'group_type': kind, 'group': value,
                               'users': int(mask.sum()), 'Recall@10': h10[mask].mean(),
                               'NDCG@10': n10[mask].mean(), 'HeadShare@10': head[top10[mask]].mean()})
    if not overall:
        raise SystemExit(f'No {split} rankings found in {out}')
    overall, groups = pd.DataFrame(overall), pd.DataFrame(groups)
    metrics = [c for c in overall.columns if c not in ('model', 'seed')]
    summary = overall.groupby('model', sort=False)[metrics].agg(['mean', 'std'])
    summary.insert(0, 'seeds', overall.groupby('model', sort=False).seed.nunique())
    gsum = (groups.groupby(['group_type', 'group', 'model'], sort=False)
            [['users', 'Recall@10', 'NDCG@10', 'HeadShare@10']].mean())

    adir = out / 'analysis'
    adir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(adir / f'{split}_per_seed.csv', index=False)
    summary.to_csv(adir / f'{split}_summary.csv')
    groups.to_csv(adir / f'{split}_groups_per_seed.csv', index=False)
    gsum.to_csv(adir / f'{split}_groups_summary.csv')
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(f'== {split}: mean (std) over seeds; head = top 20% items ({head.sum()} items) ==')
    print(summary.to_string())
    print('\n== per group (mean over seeds) ==')
    print(gsum.unstack('model')['NDCG@10'].to_string())
    print(f'\nwritten to {adir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['movielens', 'beauty'], default=None)
    ap.add_argument('--data', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--split', choices=['validation', 'test'], default='test')
    a = ap.parse_args()
    beauty = a.dataset == 'beauty'
    main(a.data or ('data/processed_beauty' if beauty else 'data/processed'),
         a.out or ('results/beauty' if beauty else 'results'), a.split)
