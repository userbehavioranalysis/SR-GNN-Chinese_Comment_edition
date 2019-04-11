# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:17:52 2019

@author: dell
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)

test = SessionGraph(opt, 310)

#测试显示会话图
#model = trans_to_cuda(SessionGraph(opt, 310))  #模型构建就靠这句话

#测试读取训练数据的图的节点
#all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
#g = build_graph(all_train_seq)
#print(len(g.node))

#测试
train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
train_data_compare = Data(train_data, shuffle=True)
slices = train_data_compare.generate_batch(100)














