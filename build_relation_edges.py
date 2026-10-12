#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the typed item-item relation edge table used for graph augmentation.

Input : the HSAL interaction csv (to recover the factorized item ids and series info) and the
        Version B LLM match files for prequel and sequel (matches_full_corpus_*.csv).
Output: Data/<out>.csv with columns
            src, dst, rtype, origin, cosine, in_series_pair, series_order_ok
        where ids are the factorized HSAL item ids (same pd.factorize order that generate_neg.py,
        sinusoidal.py and new_data1.py use) and the edge is directed src -> dst.

Relation types (message flows src -> dst, so dst aggregates from src):
        rtype 0 : src is a PREQUEL of dst
        rtype 1 : src is a SEQUEL  of dst
An LLM answer "matched book m is the prequel of central book c" gives (m -> c, 0) and its inverse
(c -> m, 1). An LLM sequel answer "m is a sequel of c" gives (m -> c, 1) and the inverse (c -> m, 0).

Filters (fixed rules, no tuning):
    1. the match must have a matched item id (cosine >= 0.5 or accepted exact title match)
    2. no self loops
    3. 'strict' also drops low-coherence matches unless the series check confirmed them
    4. direction check: if both items share a series id and the stated direction contradicts the
       series position order, the edge is dropped
    5. if the same ordered pair ends up with both rtype 0 and rtype 1, both are dropped
"""
import argparse
import re
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Goodreads_young_adult_HSAL_100k')
parser.add_argument('--prequel', required=True)
parser.add_argument('--sequel', required=True)
parser.add_argument('--filter', default='strict', choices=['all', 'strict', 'confirmed'])
parser.add_argument('--out', default=None)
opt = parser.parse_args()

inter = pd.read_csv('./Data/' + opt.data + '.csv')
raw_ids = inter['item_id'].values
codes, uniques = pd.factorize(inter['item_id'])      # same order as generate_neg.py / new_data1.py
book2hsal = {int(b): int(i) for i, b in enumerate(uniques)}

# series id and position per factorized id
ser = inter.assign(hid=codes).drop_duplicates('hid').set_index('hid')['item_series_id']
sid, pos = {}, {}
for h, s in ser.items():
    m = re.match(r'Series_(\w+?)_P([\d.]+)$', str(s))
    if m:
        sid[h], pos[h] = m.group(1), float(m.group(2))


def load(path, kind):
    d = pd.read_csv(path)
    n0 = len(d)
    d = d[d['matched_item_id'].notna()].copy()
    d['central'] = d['item_id'].astype(int).map(book2hsal)
    d['matched'] = d['matched_item_id'].astype(int).map(book2hsal)
    d = d.dropna(subset=['central', 'matched'])
    d['central'] = d['central'].astype(int)
    d['matched'] = d['matched'].astype(int)
    d['kind'] = kind
    return d, n0


stats = {'filter': opt.filter}
frames = []
for path, kind in ((opt.prequel, 'prequel'), (opt.sequel, 'sequel')):
    d, n0 = load(path, kind)
    stats[kind + '_rows'] = n0
    stats[kind + '_matched'] = len(d)
    frames.append(d)
M = pd.concat(frames, ignore_index=True)

M = M[M['central'] != M['matched']]
stats['after_self_loop'] = len(M)

if opt.filter == 'strict':
    keep = (M['low_coherence_flag'].astype(str) != 'True') | (M['series_match'] == 'match')
    M = M[keep]
elif opt.filter == 'confirmed':
    M = M[(M['match_type'] == 'exact') | (M['series_match'] == 'match')]
stats['after_filter'] = len(M)


def order_ok(row):
    """None if the pair is not a same-series pair, else whether the stated direction fits the series order."""
    c, m = row['central'], row['matched']
    if c in sid and m in sid and sid[c] == sid[m]:
        if row['kind'] == 'prequel':
            return pos[m] < pos[c]
        return pos[m] > pos[c]
    return None


M['series_order_ok'] = M.apply(order_ok, axis=1)
stats['same_series_pairs'] = int(M['series_order_ok'].notna().sum())
stats['same_series_direction_wrong_dropped'] = int((M['series_order_ok'] == False).sum())
M = M[M['series_order_ok'] != False]

rows = []
for _, r in M.iterrows():
    c, m, k = int(r['central']), int(r['matched']), r['kind']
    t_m_to_c = 0 if k == 'prequel' else 1       # type of the edge m -> c
    t_c_to_m = 1 - t_m_to_c                      # inverse
    cos = float(r['top1_score']) if pd.notna(r.get('top1_score')) else 1.0
    in_series = r['series_order_ok'] is True or r['series_order_ok'] == True
    rows.append((m, c, t_m_to_c, k, cos, int(in_series)))
    rows.append((c, m, t_c_to_m, k + '_inv', cos, int(in_series)))
E = pd.DataFrame(rows, columns=['src', 'dst', 'rtype', 'origin', 'cosine', 'in_series_pair'])

# conflicting types on one ordered pair -> drop both
conf = E.groupby(['src', 'dst'])['rtype'].nunique()
bad = conf[conf > 1].index
stats['conflicting_pairs_dropped'] = len(bad)
if len(bad):
    bad_set = set(bad)
    E = E[[ (s, d) not in bad_set for s, d in zip(E['src'], E['dst'])]]

before = len(E)
E = E.sort_values('cosine', ascending=False).drop_duplicates(['src', 'dst', 'rtype']).reset_index(drop=True)
stats['duplicates_removed'] = before - len(E)
stats['final_edges'] = len(E)
stats['final_edges_by_rtype'] = {int(k): int(v) for k, v in E['rtype'].value_counts().items()}
stats['num_items'] = int(len(uniques))
stats['items_with_out_edge'] = int(E['src'].nunique())
stats['items_with_in_edge'] = int(E['dst'].nunique())
stats['max_in_degree'] = int(E.groupby('dst').size().max())
stats['mean_in_degree'] = float(E.groupby('dst').size().mean())

out = opt.out or ('Data/' + opt.data + '_relation_edges_' + opt.filter + '.csv')
E.to_csv(out, index=False)
with open(out.replace('.csv', '_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
print('saved', out)
