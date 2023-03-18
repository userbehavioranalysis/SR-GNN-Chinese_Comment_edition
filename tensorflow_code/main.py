#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 5:40
# @Author : {ZM7}
# @File : main.py
# @Software: PyCharm

# 引入 division 模块，将整数除法转换为浮点数除法
from __future__ import division
import numpy as np
# 导入模型和工具函数
from model import *
from utils import build_graph, Data, split_validation
import pickle
import argparse
import datetime

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
opt = parser.parse_args()

# 加载训练集和测试集
train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

# 加载所有训练序列数据
# all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))

# 根据不同的数据集设定节点数量
if opt.dataset == 'diginetica':
    n_node = 43098
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37484
else:
    n_node = 310

# 构建训练数据和测试数据的 Data 对象
train_data = Data(train_data, sub_graph=True, method=opt.method, shuffle=True)
test_data = Data(test_data, sub_graph=True, method=opt.method, shuffle=False)

# 初始化模型对象
model = GGNN(hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,
             lr=opt.lr, l2=opt.l2,  step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize, lr_dc=opt.lr_dc,
             nonhybrid=opt.nonhybrid)

# 打印参数
print(opt)

# 初始化最佳结果列表
best_result = [0, 0]
best_epoch = [0, 0]
# 对模型进行多轮训练和测试
for epoch in range(opt.epoch):
    print('epoch: ', epoch, '===========================================')
    # 生成训练数据的batch
    slices = train_data.generate_batch(model.batch_size)
    fetches = [model.opt, model.loss_train, model.global_step]
    print('start training: ', datetime.datetime.now())
    loss_ = []
    # 遍历所有训练数据batch进行训练
    for i, j in zip(slices, np.arange(len(slices))):
        # 获取当前训练数据batch中的数据
        adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
        # 进行一次训练
        _, loss, _ = model.run(fetches, targets, item, adj_in, adj_out, alias,  mask)
        loss_.append(loss)
    # 计算当前epoch的平均训练loss
    loss = np.mean(loss_)
    
    # 生成测试数据的batch
    slices = test_data.generate_batch(model.batch_size)
    print('start predicting: ', datetime.datetime.now())
    hit, mrr, test_loss_ = [], [],[]
    # 遍历所有测试数据batch进行预测
    for i, j in zip(slices, np.arange(len(slices))):
        # 获取当前测试数据batch中的数据
        adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)
        # 进行一次预测
        scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, adj_in, adj_out, alias,  mask)
        # 保存当前测试batch的loss
        test_loss_.append(test_loss)
        # 对每个用户的推荐结果进行评估
        index = np.argsort(scores, 1)[:, -20:]
        for score, target in zip(index, targets):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (20-np.where(score == target - 1)[0][0]))
    # 计算当前epoch的平均测试loss、Recall@20和MMR@20
    hit = np.mean(hit)*100
    mrr = np.mean(mrr)*100
    test_loss = np.mean(test_loss_)
    # 更新最佳结果
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1]=epoch
    # 打印当前epoch的训练和测试结果
    print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'%
          (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))

