#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Control edge tables for the graph-augmentation study (same size as the real table).
  random : every dst keeps its in-degree and relation types, the src items are replaced by items drawn
           uniformly from all items (no self loops, no duplicate pairs).
  shuffle: the real (src, dst) pairs are kept, the relation types are permuted across edges.
A gain of the real table over both controls is the evidence that the LLM relations, and not just extra
edges or extra nodes, help.
"""
import argparse
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--data', default='Goodreads_young_adult_HSAL_100k')
p.add_argument('--edges', required=True)
p.add_argument('--seed', type=int, default=0)
opt = p.parse_args()

rng = np.random.RandomState(opt.seed)
E = pd.read_csv(opt.edges)
n_items = pd.read_csv('./Data/' + opt.data + '.csv')['item_id'].nunique()
base = opt.edges.replace('.csv', '')

S = E.copy()
S['rtype'] = rng.permutation(E['rtype'].values)
S.to_csv(base + '_shuffle.csv', index=False)

R = E.copy()
used = set()
new_src = []
for d, s in zip(E['dst'].values, E['src'].values):
    while True:
        c = int(rng.randint(0, n_items))
        if c != d and (c, d) not in used:
            used.add((c, d))
            break
    new_src.append(c)
R['src'] = new_src
R['in_series_pair'] = 0
R['cosine'] = 0.0
R.to_csv(base + '_random.csv', index=False)
print('items', n_items, 'edges', len(E), '->', base + '_shuffle.csv', base + '_random.csv')
