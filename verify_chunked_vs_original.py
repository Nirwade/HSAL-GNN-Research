#!/usr/bin/env python
# Check that the chunked pipeline reproduces the original single-CSV pipeline.
#
# Run the original on the full CSV (generate_neg.py, sinusoidal.py, new_data1.py --data <orig>)
# and the chunked scripts on its chunks (--data <name>), then compare. IDs are numbered differently
# in the two pipelines, so everything is mapped back to the raw CSV ids before comparing.
import argparse
import glob
import os
import pickle
import random

import dgl
import numpy as np
import pandas as pd

from new_data1_chunked import load_id_map, state_dir


def original_maps(orig):
    data = pd.read_csv(f'Data/{orig}.csv', dtype={'user_id': str, 'item_id': str})
    users = pd.factorize(data['user_id'])[1]
    items = pd.factorize(data['item_id'])[1]
    return np.asarray(users), np.asarray(items)


def chunked_maps(name):
    users = load_id_map(os.path.join(state_dir(name), 'users.csv'))
    books = load_id_map(os.path.join(state_dir(name), 'books.csv'))
    u = np.empty(len(users), dtype=object)
    b = np.empty(len(books), dtype=object)
    for raw, i in users.items():
        u[i] = raw
    for raw, i in books.items():
        b[i] = raw
    return u, b


def raw_edges(graph, users, items):
    src, dst = graph.edges(etype='by')
    item_ids = graph.nodes['item'].data['item_id'][src].numpy()
    user_ids = graph.nodes['user'].data['user_id'][dst].numpy()
    time = graph.edges['by'].data['time'].numpy()
    return sorted(zip(users[user_ids], items[item_ids], time.tolist()))


def sample_index(base, users):
    out = {}
    for split in ['train', 'val', 'test']:
        for path in glob.glob(f'{base}/{split}/*/*.bin'):
            uid, j = os.path.basename(path)[:-4].split('_')
            out[(split, users[int(uid)], int(j))] = path
    return out


def describe_sample(path, users, items):
    graphs, labels = dgl.load_graphs(path)
    return (items[int(labels['target'][0])], tuple(raw_edges(graphs[0], users, items)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig', default='Goodreads_young_adult_HSAL_100k')
    parser.add_argument('--data', default='Goodreads_young_adult_HSAL_100k_chunked')
    parser.add_argument('--item_max_length', type=int, default=50)
    parser.add_argument('--user_max_length', type=int, default=50)
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--n_check', type=int, default=2000, help='samples to compare edge by edge')
    opt = parser.parse_args()

    ou, oi = original_maps(opt.orig)
    cu, ci = chunked_maps(opt.data)
    print(f'users: original {len(ou):,}  chunked {len(cu):,}  same set: {set(ou) == set(cu)}')
    print(f'books: original {len(oi):,}  chunked {len(ci):,}  same set: {set(oi) == set(ci)}')

    og = dgl.load_graphs(f'Data/{opt.orig}_graph')[0][0]
    cg = dgl.load_graphs(f'Data/{opt.data}_graph')[0][0]
    same_graph = raw_edges(og, ou, oi) == raw_edges(cg, cu, ci)
    print(f'graph edges: original {og.num_edges("by"):,}  chunked {cg.num_edges("by"):,}  '
          f'identical (user, book, time) edges: {same_graph}')

    suffix = f'{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}'
    os_idx = sample_index(f'Newdata/{opt.orig}_{suffix}', ou)
    cs_idx = sample_index(f'Newdata/{opt.data}_{suffix}', cu)
    for split in ['train', 'val', 'test']:
        o = sum(1 for k in os_idx if k[0] == split)
        c = sum(1 for k in cs_idx if k[0] == split)
        print(f'{split} samples: original {o:,}  chunked {c:,}')
    print(f'same (user, position) samples: {set(os_idx) == set(cs_idx)}')

    keys = sorted(set(os_idx) & set(cs_idx))
    random.seed(0)
    keys = random.sample(keys, min(opt.n_check, len(keys)))
    same_target = same_sub = 0
    for k in keys:
        ot, oe = describe_sample(os_idx[k], ou, oi)
        ct, ce = describe_sample(cs_idx[k], cu, ci)
        same_target += ot == ct
        same_sub += oe == ce
    print(f'checked {len(keys):,} samples: same target {same_target:,}  identical subgraph edges {same_sub:,}')

    with open(f'{opt.orig}_neg', 'rb') as f:
        on = pickle.load(f)
    with open(f'{opt.data}_neg', 'rb') as f:
        cn = pickle.load(f)
    raw_to_c = {raw: i for i, raw in enumerate(cu)}
    users_checked = random.sample(range(len(ou)), min(500, len(ou)))
    same_neg = sum(set(oi[on[u]]) == set(ci[cn[raw_to_c[ou[u]]]]) for u in users_checked)
    print(f'negatives: identical unread-book sets for {same_neg}/{len(users_checked)} sampled users')

    ot = pd.read_csv(f'Series_table_sinusodial_Series_table_{opt.orig}.csv')
    ct = pd.read_csv(f'Series_table_sinusodial_Series_table_{opt.data}.csv')
    ot_set = {(oi[r.item_id], r.series_id, r.position_in_series) for r in ot.itertuples()}
    ct_set = {(ci[r.item_id], r.series_id, r.position_in_series) for r in ct.itertuples()}
    print(f'series table: original {len(ot):,} rows  chunked {len(ct):,} rows  identical: {ot_set == ct_set}')
