#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test for the relates_to graph augmentation. Run it inside the container on CCR, from the repo folder:

    python test_dgl_relation.py

It checks, on a small synthetic dataset and with the real code paths:
  1. new_data1.generate_user with rel=None and with rel=<RelationIndex> (the DGL behaviour that was unverified:
     edge_subgraph, select_topk and the rebuilt graph with the extra edge type);
  2. the original local indices (u_alis, last_alis) and the original item nodes are unchanged by the augmentation;
  3. extra nodes are appended at the end, carry item_id and dgl.NID, and respect the budget;
  4. save_graphs / load_graphs round trip, and dgl.batch with graphs that have and do not have relation edges;
  5. the HSAL forward and backward pass with use_rel, series_mode='bidir', series_typed on the batched graphs,
     and that the original model (all options off) still runs on graphs without relation edges.
Every check prints PASS or FAIL. The script exits with a non-zero code if any check fails.
"""
import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import dgl
from dgl import save_graphs, load_graphs

import new_data1
from relation_aug import RelationIndex, add_relation_edges

FAILS = []


def check(name, cond, extra=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + ((' | ' + str(extra)) if extra != '' else ''))
    if not cond:
        FAILS.append(name)


print('dgl', dgl.__version__, 'torch', torch.__version__)
rng = np.random.RandomState(0)
work = tempfile.mkdtemp(prefix='relaug_test_')
cwd0 = os.getcwd()
os.chdir(work)

# ---------- synthetic data: 40 users, 25 items, time strictly increasing per user ----------
n_users, n_items = 40, 25
rows = []
for u in range(n_users):
    L = rng.randint(5, 11)
    items = rng.choice(n_items, L, replace=False)
    for t, it in enumerate(items):
        rows.append((u, int(it), 1000 + 10 * t + u))
data = pd.DataFrame(rows, columns=['user_id', 'item_id', 'time'])
data['user_id'] = pd.factorize(data['user_id'])[0]
data['item_id'] = pd.factorize(data['item_id'])[0]
data = data.groupby('user_id').apply(new_data1.refine_time).reset_index(drop=True)
data['time'] = data['time'].astype('int64')
n_items_real = data['item_id'].nunique()

# relation edges: random but valid ids; both relation types
E = []
for _ in range(120):
    s, d = rng.randint(0, n_items_real, 2)
    if s != d:
        E.append((int(s), int(d), int(rng.randint(0, 2)), float(rng.rand()), int(rng.randint(0, 2))))
E = pd.DataFrame(E, columns=['src', 'dst', 'rtype', 'cosine', 'in_series_pair']).drop_duplicates(['src', 'dst', 'rtype'])
E.to_csv('edges.csv', index=False)
rel = RelationIndex('edges.csv')
check('RelationIndex loaded', rel.num_edges == len(E), rel.num_edges)

graph = new_data1.generate_graph(data)
check('base graph has only by/pby', sorted(graph.etypes) == ['by', 'pby'], graph.etypes)

# ---------- 1/2/3: generate_user without and with relation edges ----------
def run(tag, rel_obj, budget):
    base = os.path.join(work, tag)
    tr, te, va = base + '/train/', base + '/test/', base + '/val/'
    res = []
    for u in sorted(data['user_id'].unique()):
        res.append(new_data1.generate_user(u, data, graph, 50, 50, tr, te, k_hop=2, val_path=va, rel=rel_obj, rel_budget=budget))
    return base, res

base0, res0 = run('plain', None, None)
base1, res1 = run('rel_b2', rel, 2)
base2, res2 = run('rel_b0', rel, 0)
base3, res3 = run('rel_unc', rel, None)
check('generate_user returns (train, test, stats)', all(len(r) == 3 for r in res1))
check('same number of train/test samples with and without relation edges',
      [r[:2] for r in res0] == [r[:2] for r in res1] == [r[:2] for r in res2])


def files(base):
    out = {}
    for sp in ('train', 'val', 'test'):
        d = os.path.join(base, sp)
        if os.path.isdir(d):
            for root, _, fs in os.walk(d):
                for f in fs:
                    out[sp + '/' + f] = os.path.join(root, f)
    return out

f0, f1, f2, f3 = files(base0), files(base1), files(base2), files(base3)
check('same file set', set(f0) == set(f1) == set(f2) == set(f3), (len(f0), len(f1)))
total_rel_edges, total_new, max_new_b2, max_new_b0 = 0, 0, 0, 0
idx_ok, order_ok, ids_ok, etype_ok, rtype_ok, nodes_ok = True, True, True, True, True, True
for name in sorted(f0):
    g0, l0 = load_graphs(f0[name])
    g1, l1 = load_graphs(f1[name])
    g0, g1 = g0[0], g1[0]
    n0 = g0.num_nodes('item')
    etype_ok &= ('relates_to' in g1.etypes) and ('relates_to' not in g0.etypes)
    idx_ok &= bool(torch.equal(l0['u_alis'], l1['u_alis']) and torch.equal(l0['last_alis'], l1['last_alis']) and torch.equal(l0['target'], l1['target']))
    order_ok &= bool(torch.equal(g0.nodes['item'].data['item_id'], g1.nodes['item'].data['item_id'][:n0]))
    order_ok &= bool(torch.equal(g0.nodes['item'].data[dgl.NID], g1.nodes['item'].data[dgl.NID][:n0]))
    ids_ok &= bool(torch.equal(g1.nodes['item'].data['item_id'], g1.nodes['item'].data[dgl.NID]))
    ids_ok &= bool(torch.equal(g0.edges['by'].data['time'], g1.edges['by'].data['time']))
    nodes_ok &= g1.num_nodes('user') == g0.num_nodes('user')
    ne = g1.num_edges('relates_to')
    total_rel_edges += ne
    total_new += g1.num_nodes('item') - n0
    rtype_ok &= bool(g1.edges['relates_to'].data['rtype'].dtype == torch.int64) and (ne == 0 or int(g1.edges['relates_to'].data['rtype'].max()) <= 1)
    g2 = load_graphs(f2[name])[0][0]
    max_new_b0 = max(max_new_b0, g2.num_nodes('item') - n0)
    max_new_b2 = max(max_new_b2, g1.num_nodes('item') - n0)
check('relates_to etype present only in augmented graphs', etype_ok)
check('u_alis / last_alis / target unchanged by augmentation', idx_ok)
check('original item nodes keep their positions (extra nodes appended at the end)', order_ok)
check('item_id == dgl.NID for all item nodes, by-edge times unchanged', ids_ok)
check('user nodes unchanged', nodes_ok)
check('rtype is int64 and in {0,1}', rtype_ok)
check('budget 2 respected', max_new_b2 <= 2, max_new_b2)
check('budget 0 adds no nodes', max_new_b0 == 0, max_new_b0)
check('some relation edges were added (test is not vacuous)', total_rel_edges > 0, total_rel_edges)
check('some extra nodes were added with budget 2', total_new > 0, total_new)
print('relation edges added in total:', total_rel_edges, ' extra nodes:', total_new)

# ---------- 4: batching graphs with and without relation edges ----------
gs = [load_graphs(f1[n])[0][0] for n in sorted(f1)[:8]]
has_edges = [g.num_edges('relates_to') for g in gs]
bg = dgl.batch(gs)
check('dgl.batch works on augmented graphs', bg.num_nodes('item') == sum(g.num_nodes('item') for g in gs))
check('batched relates_to edges add up', bg.num_edges('relates_to') == sum(has_edges), has_edges)
check('batch_num_nodes(item) includes extra nodes', int(bg.batch_num_nodes('item').sum()) == bg.num_nodes('item'))

# ---------- 5: model forward / backward ----------
# tiny series table in the file name HSALLayers expects
data_name = 'relaug_smoke'
emb_dim = 50
srows = []
for it in range(n_items_real):
    series = 'Series_%d' % (it // 4)
    pos = it % 4 + 1
    pe = [float(np.sin(pos / (10000 ** (2 * (i // 2) / emb_dim)))) if i % 2 == 0 else float(np.cos(pos / (10000 ** (2 * (i // 2) / emb_dim)))) for i in range(emb_dim)]
    srows.append((it, 'Series_x_P%d' % pos, series, pos, str(pe)))
pd.DataFrame(srows, columns=['item_id', 'item_series_id', 'series_id', 'position_in_series', 'positional_embedding']).to_csv(
    'Series_table_sinusodial_Series_table_%s.csv' % data_name, index=False)

from HSAL_for_BPR1 import HSAL, collate_test


def forward_backward(graphs, **kw):
    model = HSAL(user_num=n_users + 10, item_num=n_items_real + 10, input_dim=emb_dim, item_max_length=50, user_max_length=50,
                 feat_drop=0.0, attn_drop=0.0, user_long='orgat', user_short='att', item_long='orgat', item_short='att',
                 user_update='rnn', item_update='rnn', last_item=True, layer_num=2, data_name=data_name, **kw)
    batch = dgl.batch(graphs)
    dev = model.device
    user = torch.stack([torch.as_tensor(g_u) for g_u in us]).long().reshape(-1)
    last = torch.stack([torch.as_tensor(l) for l in lasts]).long().reshape(-1)
    pos = torch.stack([torch.as_tensor(t) for t in tgts]).long().reshape(-1)
    neg = torch.randint(0, n_items_real, (len(graphs),))
    ps, ns = model(batch, user.to(dev), last.to(dev), pos_tar=pos.to(dev), neg_tar=neg.to(dev), is_training=True)
    loss = -torch.nn.functional.logsigmoid(ps - ns).mean()
    loss.backward()
    return model, loss


names8 = sorted(f1)[:8]
loaded = [load_graphs(f1[n]) for n in names8]
gs = [x[0][0] for x in loaded]
us = [x[1]['u_alis'] for x in loaded]
lasts = [x[1]['last_alis'] for x in loaded]
tgts = [x[1]['target'] for x in loaded]
try:
    m, loss = forward_backward(gs, use_rel=True, series_mode='bidir', series_typed=True)
    check('forward/backward with use_rel + bidir + typed series', bool(torch.isfinite(loss)), float(loss))
    g_rel = m.layers[0].rel_type_embedding.weight.grad
    check('gradient reaches the relation type embedding', g_rel is not None and float(g_rel.abs().sum()) > 0)
    g_dir = m.layers[0].series_dir_embedding.weight.grad
    check('gradient reaches the series direction embedding', g_dir is not None)
except Exception as e:
    import traceback
    traceback.print_exc()
    check('forward/backward with use_rel + bidir + typed series', False, repr(e))

loaded0 = [load_graphs(f0[n]) for n in names8]
gs0 = [x[0][0] for x in loaded0]
try:
    m0, loss0 = forward_backward(gs0, use_rel=False, series_mode='forward', series_typed=False)
    check('original model (all options off) runs on graphs without relation edges', bool(torch.isfinite(loss0)), float(loss0))
    check('original model has no extra parameters', not hasattr(m0.layers[0], 'rel_type_embedding') and not hasattr(m0.layers[0], 'series_dir_embedding'))
except Exception as e:
    import traceback
    traceback.print_exc()
    check('original model runs on graphs without relation edges', False, repr(e))

try:
    m2, loss2 = forward_backward(gs, use_rel=False, series_mode='forward', series_typed=False)
    check('original model also ignores relates_to edges when use_rel is off', bool(torch.isfinite(loss2)), float(loss2))
except Exception as e:
    check('original model ignores relates_to edges when use_rel is off', False, repr(e))

# eval path (collate_test style scoring with negatives)
try:
    m.eval()
    with torch.no_grad():
        batch = dgl.batch(gs)
        dev = m.device
        user = torch.stack([torch.as_tensor(u) for u in us]).long().reshape(-1)
        last = torch.stack([torch.as_tensor(l) for l in lasts]).long().reshape(-1)
        neg = torch.randint(0, n_items_real, (len(gs), 20))
        label = torch.stack([torch.as_tensor(t) for t in tgts]).long().reshape(-1)
        score, top = m(batch, user.to(dev), last.to(dev), neg_tar=torch.cat([label.unsqueeze(1), neg], -1).to(dev), is_training=False)
    check('evaluation forward works', top.shape == (len(gs), 21), tuple(top.shape))
except Exception as e:
    import traceback
    traceback.print_exc()
    check('evaluation forward works', False, repr(e))

os.chdir(cwd0)
shutil.rmtree(work, ignore_errors=True)
print('\n%d check(s) failed' % len(FAILS) if FAILS else '\nALL CHECKS PASSED')
sys.exit(1 if FAILS else 0)
