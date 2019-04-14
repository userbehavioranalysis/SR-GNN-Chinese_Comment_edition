#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes():
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail): #输入：将所有输入序列all_usr_pois, 末尾补全的数据item_tail
    us_lens = [len(upois) for upois in all_usr_pois] #每一个输入序列的长度的列表
    len_max = max(us_lens)  #得到输入序列的最大长度
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)] #将所有输入序列按照最长长度尾部补全 item_tail
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] #有序列的位置是[1],没有动作序列的位置是[0]
    return us_pois, us_msks, len_max  #输出：补全0后的序列us_pois, 面罩序列us_msks, 最大序列长度len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]   #输入序列的列表
        inputs, mask, len_max = data_masks(inputs, [0])  #详见函数 ---> data_masks() 这个函数使得所有会话按照最长的长度补0了！
        self.inputs = np.asarray(inputs)  #补全0后的输入序列，并转化成array()
        self.mask = np.asarray(mask)     #面罩序列，并转化成array()
        self.len_max = len_max    #最大序列长度
        self.targets = np.asarray(data[1])  #预测的序列的列表
        self.length = len(inputs)  #输入样本的大小
        self.shuffle = shuffle   #是否打乱数据
        self.graph = graph    #数据图 (?) 这个似乎没有用到

    def generate_batch(self, batch_size):  #根据批的大小生成批数据的索引，如果shuffle则打乱数据
        if self.shuffle:  #如果需要打乱数据
            shuffled_arg = np.arange(self.length)  #生成array([0,1,...,样本长度-1])
            np.random.shuffle(shuffled_arg)  #随机打乱shuffled_arg的顺序
            self.inputs = self.inputs[shuffled_arg]  #按照shuffled_arg来索引输入数据
            self.mask = self.mask[shuffled_arg]   #按照shuffled_arg来索引面罩数据
            self.targets = self.targets[shuffled_arg]  #按照shuffled_arg来索引预测目标数据
        n_batch = int(self.length / batch_size)  #得到训练批数
        if self.length % batch_size != 0: #批数需要取向上取整
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)  #所有数据按照批进行拆分。eg:[0,..,99][100,..,199]...
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]  #最后一批有多少给多少。eg:[500,..506]
        return slices

    def get_slice(self, i):  #根据索引i得到对应的数据
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i] #得到对应索引的输入，面罩，目标数据
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))  #n_node存储每个输入序列单独出现的点击动作类别的个数的列表
        max_n_node = np.max(n_node)   #得到批最长唯一动作会话序列的长度
        for u_input in inputs:  # u_input 为一个会话序列
            node = np.unique(u_input)  #该循环的会话的唯一动作序列
            items.append(node.tolist() + (max_n_node - len(node)) * [0])  #单个点击动作序列的唯一类别并按照批最大类别补全0
            u_A = np.zeros((max_n_node, max_n_node))  #存储行为矩阵的二维向量(方阵)，长度是最大唯一动作的数量
            for i in np.arange(len(u_input) - 1):  #循环该序列的长度
                if u_input[i + 1] == 0:  #循环到i的下一个动作时“0”动作时退出循环，因为0代表序列已经结束，后面都是补的动作0
                    break
                u = np.where(node == u_input[i])[0][0]  #该动作对应唯一动作集合的序号
                v = np.where(node == u_input[i + 1])[0][0] #下一个动作对应唯一动作集合的序号
                u_A[u][v] = 1  #前一个动作u_input[i]转移到后一个动作u_input[i + 1]的次数变成1
            u_sum_in = np.sum(u_A, 0) #矩阵列求和，最后变成一行
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1) #矩阵行求和，最后变成一列
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()  #得到一个会话的连接矩阵
            A.append(u_A)  #存储该批数据图矩阵的列表，u_A方阵的长度相同——为该批最长唯一动作会话序列的长度
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) #动作序列对应唯一动作集合的位置
        return alias_inputs, A, items, mask, targets
        #返回：动作序列对应唯一动作集合的位置，该批数据图矩阵的列表，单个点击动作序列的唯一类别并按照批最大类别补全0列表，面罩，目标数据
    
