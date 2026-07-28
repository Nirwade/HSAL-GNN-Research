#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/1/31 11:15
# @Author : ZM7
# @File : generate_neg
# @Software: PyCharm
import argparse
import pandas as pd
import pickle
from utils import myFloder, pickle_loader, collate, trans_to_cuda, eval_metric, collate_test, user_neg


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Goodreads_young_adult_HSAL_100k', help='dataset name')
opt = parser.parse_args()
dataset = opt.data
data = pd.read_csv('./Data/' + dataset + '.csv')
data['user_id'] = pd.factorize(data['user_id'])[0]
data['item_id'] = pd.factorize(data['item_id'])[0]
user = data['user_id'].unique()
item = data['item_id'].unique()
user_num = len(user)
item_num = len(item)

data_neg = user_neg(data, item_num)
print('data_neg',data_neg)

f = open(dataset+'_neg', 'wb')
pickle.dump(data_neg,f)
f.close()

print('generate_neg.py ran successfully')
