#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph augmentation with typed item-item relation edges.

The sampled training graph (built in new_data1.py) only has the edge types 'by' (item -> user) and
'pby' (user -> item). This module adds a third edge type ('item', 'relates_to', 'item') to a sampled
graph, with one integer relation type per edge stored in edge data 'rtype'
(0: src is a prequel of dst, 1: src is a sequel of dst), and, when needed, extra item nodes for
relation neighbours that are not in the sample (one-hop expansion).

Design rules (see HSAL_Graph_Augmentation_Plan_and_Decisions.md):
  * the base graph and the existing select_topk sampling are untouched; relation edges are added
    after sampling, so nothing about the original sampling changes;
  * the dst of a relation edge must already be an item of the user's recent sequence that is in the
    sample ("a reliable relationship from an item already relevant to the current sample");
  * a relation neighbour that is already in the sample is free; a neighbour that is not in the
    sample becomes an extra item node, and the number of extra nodes per sample is capped by `budget`;
  * extra nodes are appended after the existing item nodes, so the local indices that new_data1.py
    stores (u_alis, last_alis) stay valid;
  * every saved graph has the 'relates_to' edge type, even when it has no relation edges, so that
    dgl.batch can batch graphs together.
"""
import numpy as np
import pandas as pd
import torch
import dgl


class RelationIndex(object):
    """dst item -> (src items, relation types), strongest match first."""

    def __init__(self, edge_csv):
        E = pd.read_csv(edge_csv)
        E = E.sort_values(['dst', 'cosine'], ascending=[True, False], kind='mergesort')
        self.by_dst = {}
        for dst, g in E.groupby('dst'):
            self.by_dst[int(dst)] = (g['src'].values.astype(np.int64), g['rtype'].values.astype(np.int64))
        self.num_edges = int(len(E))

    def neighbours(self, dst):
        return self.by_dst.get(int(dst))


def add_relation_edges(g, rel, dst_candidates, budget=None):
    """Return (new_graph, info).

    g              sampled graph with etypes 'by' and 'pby'; item nodes carry 'item_id' (global
                   factorized id) and dgl.NID
    rel            RelationIndex
    dst_candidates global item ids of the user's recent items, most recent first
    budget         max number of extra item nodes added to this sample (None = no cap, 0 = only edges
                   between items that are already in the sample)
    """
    item_gid = g.nodes['item'].data['item_id'].cpu().numpy()
    n_user = g.num_nodes('user')
    n_item = g.num_nodes('item')
    local_of = {int(gid): i for i, gid in enumerate(item_gid)}

    src_l, dst_l, rt_l, new_gids = [], [], [], []
    seen_dst = set()
    skipped_budget = 0
    for d in dst_candidates:
        d = int(d)
        if d in seen_dst or d not in local_of:
            continue
        seen_dst.add(d)
        nb = rel.neighbours(d)
        if nb is None:
            continue
        for s, t in zip(nb[0], nb[1]):
            s = int(s)
            if s in local_of:
                ls = local_of[s]
            else:
                if budget is not None and len(new_gids) >= budget:
                    skipped_budget += 1
                    continue
                ls = n_item + len(new_gids)
                local_of[s] = ls
                new_gids.append(s)
            src_l.append(ls)
            dst_l.append(local_of[d])
            rt_l.append(int(t))

    n_new = len(new_gids)
    by = g.edges(etype='by')
    pby = g.edges(etype='pby')
    rel_edges = (torch.tensor(src_l, dtype=torch.int64), torch.tensor(dst_l, dtype=torch.int64))
    ng = dgl.heterograph({('item', 'by', 'user'): by,
                          ('user', 'pby', 'item'): pby,
                          ('item', 'relates_to', 'item'): rel_edges},
                         num_nodes_dict={'user': n_user, 'item': n_item + n_new})

    # node data: copy, and extend the item data for the extra nodes
    for key, val in g.nodes['user'].data.items():
        ng.nodes['user'].data[key] = val
    new_gid_t = torch.tensor(new_gids, dtype=torch.int64)
    for key, val in g.nodes['item'].data.items():
        if key == 'item_id' or key == dgl.NID:
            extra = new_gid_t.to(val.dtype)
        else:
            extra = torch.zeros((n_new,) + tuple(val.shape[1:]), dtype=val.dtype)
        ng.nodes['item'].data[key] = torch.cat([val, extra], 0)
    # edge data of the original edge types
    for et in ('by', 'pby'):
        for key, val in g.edges[et].data.items():
            ng.edges[et].data[key] = val
    ng.edges['relates_to'].data['rtype'] = torch.tensor(rt_l, dtype=torch.int64)

    info = {'rel_edges': len(src_l), 'new_nodes': n_new, 'skipped_budget': skipped_budget,
            'item_nodes_before': n_item, 'new_gids': new_gids,
            'rel_src_gids': [int(item_gid[s]) if s < n_item else new_gids[s - n_item] for s in src_l],
            'rel_dst_gids': [int(item_gid[d]) for d in dst_l]}
    return ng, info
