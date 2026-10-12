#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm

import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import sample_neighbors, select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed
from relation_aug import RelationIndex, add_relation_edges

# 计算item序列的相对次序
def cal_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['order'] = range(len(data))
    return data

# 计算user序列的相对次序
def cal_u_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['u_order'] = range(len(data))
    return data

def refine_time(data):
    data = data.sort_values(['time'], kind='mergesort')
    time_seq = data['time'].values
    time_gap = 1
    for i, da in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i+1] or time_seq[i] > time_seq[i+1]:
            time_seq[i+1] = time_seq[i+1] + time_gap
            time_gap += 1
    data['time'] = time_seq
    return  data

def generate_graph(data):
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    data = data.groupby('user_id').apply(cal_order).reset_index(drop=True)
    data = data.groupby('item_id').apply(cal_u_order).reset_index(drop=True)
    user = data['user_id'].values
    item = data['item_id'].values
    time = data['time'].values
    graph_data = {('item','by','user'):(torch.tensor(item), torch.tensor(user)),
                  ('user','pby','item'):(torch.tensor(user), torch.tensor(item))}
    graph = dgl.heterograph(graph_data)
    graph.edges['by'].data['time'] = torch.LongTensor(time)
    graph.edges['pby'].data['time'] = torch.LongTensor(time)
    #graph.edges['by'].data['t'] = torch.tensor(data['order'])
    # graph.edges['by'].data['rt'] = torch.tensor(data['re_order'])
    # graph.edges['pby'].data['t'] = torch.tensor(data['u_order'])
    #graph.edges['pby'].data['rt'] = torch.tensor(data['u_re_order'])
    graph.nodes['user'].data['user_id'] = torch.LongTensor(np.unique(user))
    graph.nodes['item'].data['item_id'] = torch.LongTensor(np.unique(item))
    # graph.nodes['item'].data['last_user'] = torch.tensor(data['u_last'])
    # graph.nodes['user'].data['last_item'] = torch.tensor(data['last'])
    return graph


def generate_user(user, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop=3, val_path=None,
                  rel=None, rel_budget=None, exclude_val=False):
    # rel: RelationIndex (relation_aug.py) or None. With rel=None the output is identical to the original code.
    # exclude_val: the original code also writes the validation example (j = n-3) into the training set.
    #   With exclude_val=True the validation example is NOT used for training, so validation is honest.
    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0
    stats = {}
    # 生成训练数据
    if len(u_seq) < 3:
        return 0, 0, stats
    else:
        for j, t  in enumerate(u_time[0:-1]):
            if j == 0:
                continue
            if j < item_max_length:
                start_t = u_time[0]
            else:
                start_t = u_time[j - item_max_length]
            sub_u_eid = (graph.edges['by'].data['time'] < u_time[j+1]) & (graph.edges['by'].data['time'] >= start_t)
            sub_i_eid = (graph.edges['pby'].data['time'] < u_time[j+1]) & (graph.edges['pby'].data['time'] >= start_t)
            sub_graph = dgl.edge_subgraph(graph, edges = {'by':sub_u_eid, 'pby':sub_i_eid}, relabel_nodes=False)
            u_temp = torch.tensor([user])
            his_user = torch.tensor([user])
            graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user':u_temp})
            i_temp = torch.unique(graph_i.edges(etype='by')[0])
            his_item = torch.unique(graph_i.edges(etype='by')[0])
            edge_i = [graph_i.edges['by'].data[dgl.NID]]
            edge_u = []
            for _ in range(k_hop-1):
                graph_u = select_topk(sub_graph, user_max_length, weight='time', nodes={'item': i_temp})  # item的邻居user
                u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)[-user_max_length:]
                #u_temp = torch.unique(torch.cat((u_temp, graph_u.edges(etype='pby')[0])))
                graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                #i_temp = torch.unique(torch.cat((i_temp, graph_i.edges(etype='by')[0])))
                i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)
                his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
                edge_i.append(graph_i.edges['by'].data[dgl.NID])
                edge_u.append(graph_u.edges['pby'].data[dgl.NID])
            all_edge_u = torch.unique(torch.cat(edge_u))
            all_edge_i = torch.unique(torch.cat(edge_i))
            fin_graph = dgl.edge_subgraph(sub_graph, edges={'by':all_edge_i,'pby':all_edge_u})
            target = u_seq[j+1]
            last_item = u_seq[j]
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==user)[0]
            last_alis = torch.where(fin_graph.nodes['item'].data['item_id']==last_item)[0]
            # 分别计算user和last_item在fin_graph中的索引
            if rel is not None:
                # one-hop expansion with typed relation edges, added after the original sampling
                recent = u_seq[max(0, j - item_max_length + 1): j + 1][::-1]   # most recent first
                fin_graph, info = add_relation_edges(fin_graph, rel, recent, budget=rel_budget)
                split_name = 'test' if j == split_point - 1 else ('val' if j == split_point - 2 else 'train')
                tgt = int(target)
                lst = int(last_item)
                pairs = list(zip(info['rel_src_gids'], info['rel_dst_gids']))
                for k_, v_ in (('samples', 1), ('rel_edges', info['rel_edges']), ('new_nodes', info['new_nodes']),
                               ('item_nodes', info['item_nodes_before']),
                               ('target_rel_of_last', int((tgt, lst) in pairs)),
                               ('target_rel_of_history', int(any(s_ == tgt for s_, _ in pairs))),
                               ('target_is_new_node', int(tgt in info['new_gids']))):
                    key = split_name + '_' + k_
                    stats[key] = stats.get(key, 0) + v_
            train_cut = split_point - 1 - (1 if exclude_val else 0)
            if j < train_cut:
                save_graphs(train_path+ '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis':u_alis, 'last_alis': last_alis})
                train_num += 1
            if j == split_point - 1 - 1:
                save_graphs(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis,
                             'last_alis': last_alis})
            if j == split_point - 1:
                save_graphs(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis':u_alis, 'last_alis': last_alis})
                test_num += 1
        return train_num, test_num, stats


def generate_data(data, graph, item_max_length, user_max_length, train_path, test_path, val_path, job=10, k_hop=3,
                  rel=None, rel_budget=None, exclude_val=False, max_users=None):
    user = data['user_id'].unique()
    if max_users is not None:
        user = user[:max_users]
    a = Parallel(n_jobs=job)(delayed(lambda u: generate_user(u, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop, val_path,
                                                             rel=rel, rel_budget=rel_budget, exclude_val=exclude_val))(u) for u in user)
    return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Goodreads_YoungAdult_HSAL_Final', help='data name: sample')
    parser.add_argument('--graph', action='store_true', help='no_batch')
    parser.add_argument('--item_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--user_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    # --- graph augmentation (all optional; defaults reproduce the original behaviour) ---
    parser.add_argument('--rel_edges', default=None, help='relation edge csv from build_relation_edges.py; enables relates_to edges')
    parser.add_argument('--rel_budget', type=int, default=-1, help='max extra item nodes per sample; -1 = no cap')
    parser.add_argument('--tag', default='', help='suffix for the Newdata folder, e.g. _rel_strict_b5')
    parser.add_argument('--exclude_val', action='store_true', help='do not train on the validation example (j = n-3)')
    parser.add_argument('--max_users', type=int, default=None, help='smoke test: only process the first N users')
    opt = parser.parse_args()
    rel = RelationIndex(opt.rel_edges) if opt.rel_edges else None
    rel_budget = None if opt.rel_budget < 0 else opt.rel_budget
    data_path = './Data/' + opt.data + '.csv'
    graph_path = './Data/' + opt.data + '_graph'
    data = pd.read_csv(data_path)

    # 2. REMAP IDs to contiguous integers (0 to N-1)
    data['user_id'] = pd.factorize(data['user_id'])[0]
    data['item_id'] = pd.factorize(data['item_id'])[0]
    
    # 3. Now apply your time refinement
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    data['time'] = data['time'].astype('int64')


    # if opt.graph:
    #     graph = generate_graph(data)
    #     save_graphs(graph_path, graph)
    # else:
    if not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
    train_path = f'Newdata/{opt.data}{opt.tag}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
    val_path = f'Newdata/{opt.data}{opt.tag}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
    test_path = f'Newdata/{opt.data}{opt.tag}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
    #generate_user(41, data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, k_hop=opt.k_hop)
    print('start:', datetime.datetime.now())
    all_num = generate_data(data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, val_path, job=opt.job, k_hop=opt.k_hop,
                            rel=rel, rel_budget=rel_budget, exclude_val=opt.exclude_val, max_users=opt.max_users)
    train_num = 0
    test_num = 0
    tot = {}
    for num_ in all_num:
        train_num += num_[0]
        test_num += num_[1]
        for k_, v_ in (num_[2] if len(num_) > 2 else {}).items():
            tot[k_] = tot.get(k_, 0) + v_
    print('The number of train set:', train_num)
    print('The number of test set:', test_num)
    if rel is not None:
        import json
        stats_path = f'Newdata/{opt.data}{opt.tag}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/augmentation_stats.json'
        with open(stats_path, 'w') as f_:
            json.dump({'rel_edges_csv': opt.rel_edges, 'rel_budget': opt.rel_budget, 'exclude_val': opt.exclude_val, 'totals': tot}, f_, indent=2)
        print('augmentation stats:', json.dumps(tot))
    print('end:', datetime.datetime.now())


