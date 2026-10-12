#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Expansion-budget analysis from the edge table and the interaction sequences only (no DGL, no model).

For every example of the chosen splits it replays the selection rule of relation_aug.add_relation_edges:
walk the user's recent items (most recent first), take the relation neighbours of each item, neighbours
that are already in the sample are free, other neighbours cost one extra node and stop being added once
the budget is used up. "In the sample" is approximated by the user's own window items, which is the part
of the sampled graph that is known without DGL, so the extra-node counts here are an upper bound. The
exact numbers come from Newdata/<...>/augmentation_stats.json written by new_data1.py.

Reported per budget B: coverage (the target is a relation neighbour of a window item and was added),
share of coverage kept compared with no cap, mean extra nodes, mean relation edges, and the share of the
added edges whose two books share a series (the series-confirmed share, a reliability proxy).

Only train and validation examples are used. The test split is never read for choosing the budget.

Selection rule, fixed in advance: the smallest budget that keeps at least 90% of the uncapped train
coverage.
"""
import argparse
import re
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Goodreads_young_adult_HSAL_100k')
parser.add_argument('--edges', required=True)
parser.add_argument('--window', type=int, default=50)
parser.add_argument('--budgets', default='0,1,2,3,5,8,-1')
parser.add_argument('--splits', default='train,val')
parser.add_argument('--out', default=None)
opt = parser.parse_args()
assert 'test' not in opt.splits.split(','), 'the test split must not be used to choose the budget'

inter = pd.read_csv('./Data/' + opt.data + '.csv')
inter['hid'] = pd.factorize(inter['item_id'])[0]
E = pd.read_csv(opt.edges).sort_values(['dst', 'cosine'], ascending=[True, False], kind='mergesort')
by_dst = {int(d): (g['src'].values, g['in_series_pair'].values) for d, g in E.groupby('dst')}

budgets = [int(b) for b in opt.budgets.split(',')]
splits = opt.splits.split(',')
inter = inter.sort_values(['user_id', 'time'], kind='stable')

rows = []   # one record per (example, budget)
for u, seq in inter.groupby('user_id')['hid']:
    seq = list(seq)
    n = len(seq)
    if n < 3:
        continue
    for j in range(1, n - 1):
        split = 'test' if j == n - 2 else ('val' if j == n - 3 else 'train')
        if split not in splits:
            continue
        target = seq[j + 1]
        recent = seq[max(0, j - opt.window + 1): j + 1][::-1]
        window = set(recent)
        for B in budgets:
            local = set(window)
            n_new, n_edges, n_series, covered = 0, 0, 0, 0
            seen = set()
            for d in recent:
                if d in seen or d not in by_dst:
                    continue
                seen.add(d)
                srcs, ser = by_dst[d]
                for s, sp in zip(srcs, ser):
                    s = int(s)
                    if s not in local:
                        if B >= 0 and n_new >= B:
                            continue
                        local.add(s)
                        n_new += 1
                    n_edges += 1
                    n_series += int(sp)
                    if s == target:
                        covered = 1
            rows.append((split, B, covered, n_new, n_edges, n_series))

R = pd.DataFrame(rows, columns=['split', 'budget', 'covered', 'new_nodes', 'edges', 'series_edges'])
T = R.groupby(['split', 'budget']).agg(examples=('covered', 'size'), covered=('covered', 'sum'),
                                       mean_new_nodes=('new_nodes', 'mean'), mean_edges=('edges', 'mean'),
                                       edges=('edges', 'sum'), series_edges=('series_edges', 'sum')).reset_index()
T['coverage_rate'] = T['covered'] / T['examples']
T['series_confirmed_share'] = T['series_edges'] / T['edges'].replace(0, np.nan)
full = T[T['budget'] == -1].set_index('split')['covered']
T['kept_vs_uncapped'] = [r.covered / full[r.split] if full[r.split] else np.nan for r in T.itertuples()]
T['budget'] = T['budget'].replace(-1, 'uncapped')
pd.set_option('display.width', 200)
print(T[['split', 'budget', 'examples', 'covered', 'coverage_rate', 'kept_vs_uncapped', 'mean_new_nodes', 'mean_edges', 'series_confirmed_share']].to_string(index=False, float_format=lambda x: '%.4f' % x))

# fixed selection rule on the train split
tr = T[T['split'] == 'train'].copy()
tr['b'] = tr['budget'].apply(lambda b: 10 ** 6 if b == 'uncapped' else int(b))
ok = tr[tr['kept_vs_uncapped'] >= 0.9].sort_values('b')
choice = ok.iloc[0]['budget'] if len(ok) else 'uncapped'
print('\nchosen budget (smallest keeping >= 90% of uncapped train coverage):', choice)
if opt.out:
    T.to_csv(opt.out, index=False)
    with open(opt.out.replace('.csv', '_choice.json'), 'w') as f:
        json.dump({'edges': opt.edges, 'splits': splits, 'chosen_budget': str(choice)}, f)
