#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):  #输入仅需确定隐状态数和步数
        super(GNN, self).__init__()
        self.step = step  #gnn前向传播的步数 default=1
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        #有关Parameter函数的解释：首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        #并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        #所以经过类型转换这个self.XX变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        #使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。——————https://www.jianshu.com/p/d8b77cc02410
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))
        #有关nn.Linear的解释：torch.nn.Linear(in_features, out_features, bias=True)，对输入数据做线性变换：y=Ax+b
        #形状：输入: (N,in_features)  输出： (N,out_features)
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        #A-->实际上是该批数据图矩阵的列表  eg:(100,5?,10?(即5?X2))
        #hidden--> eg(100-batch_size,5?,100-embeding_size) 
        #后面所有的5?代表这个维的长度是该批唯一最大类别长度(类别数目不足该长度的会话补零)，根据不同批会变化
        #有关matmul的解释：矩阵相乘，多维会广播相乘  
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah   #input_in-->(100,5?,100)
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah  #input_out-->(100,5?,100)
        #在第2个轴将tensor连接起来
        inputs = torch.cat([input_in, input_out], 2)  #inputs-->(100,5?,200)
        #关于functional.linear(input, weight, bias=None)的解释：y= xA^T + b 应用线性变换，返回Output: (N,∗,out_features)
        #[*代表任意其他的东西]
        gi = F.linear(inputs, self.w_ih, self.b_ih) #gi-->(100,5?,300)
        gh = F.linear(hidden, self.w_hh, self.b_hh) #gh-->(100,5?,300)
        #torch.chunk(tensor, chunks, dim=0)：将tensor拆分成指定数量的块，比如下面就是沿着第2个轴拆分成3块
        i_r, i_i, i_n = gi.chunk(3, 2)  #三个都是(100,5?,100)
        h_r, h_i, h_n = gh.chunk(3, 2)  #三个都是(100,5?,100)
        resetgate = torch.sigmoid(i_r + h_r)   #resetgate-->(100,5?,100)      原文公式(3)
        inputgate = torch.sigmoid(i_i + h_i)   #inputgate-->(100,5?,100)
        newgate = torch.tanh(i_n + resetgate * h_n)  #newgate-->(100,5?,100)  原文公式(4)
        hy = newgate + inputgate * (hidden - newgate)   #hy-->(100,5?,100)    原文公式(5)
        return hy

    def forward(self, A, hidden): 
        #A-->实际上是该批数据图矩阵的列表 eg:(100,5?,10?(即5?X2)) 5?代表这个维的长度是该批唯一最大类别长度(类别数目不足该长度的会话补零)，根据不同批会变化
        #hidden--> eg:(100-batch_size,5?,100-embeding_size) 即数据图中节点类别对应低维嵌入的表示
        for i in range(self.step):  
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node): #opt-->可控输入参数, n_node-->嵌入层图的节点数目
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize  #opt.hiddenSize-->hidden state size
        self.n_node = n_node
        self.batch_size = opt.batchSize   #opt.batch_siza-->input batch size *default=100
        self.nonhybrid = opt.nonhybrid   #opt.nonhybrid-->only use the global preference to predicts
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step) #opt.step-->gnn propogation steps
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()  #交叉熵损失
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2) #Adam优化算法
        #StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1) 将每个参数组的学习率设置为每个step_size epoch
        #由gamma衰减的初始lr。当last_epoch=-1时，将初始lr设置为lr。
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()   #初始化权重参数

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        #hidden-->(100,16?,100) 其中16?代表该样本所有数据最长会话的长度(不同数据集会不同)，单个样本其余部分补了0
        #mask-->(100,16?) 有序列的位置是[1],没有动作序列的位置是[0]
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size(100,100) 这是最后一个动作对应的位置，即文章中说的局部偏好
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size(100,1,100) 局部偏好线性变换后改成能计算的维度
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size (100,16?,100) 即全局偏好
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  #(100,16,1)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1) #(100,100)  原文中公式(6)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))  #原文中公式(7)
        b = self.embedding.weight[1:]  # n_nodes x latent_size  (309,100)
        scores = torch.matmul(a, b.transpose(1, 0))   #原文中公式(8)
        return scores  #(100,309)

    def forward(self, inputs, A):  
        #inputs-->单个点击动作序列的唯一类别并按照批最大唯一类别长度补全0列表(即图矩阵的元素的类别标签列表)  A-->实际上是该批数据图矩阵的列表
#        print(inputs.size())  #测试打印下输入的维度  （100-batch_size,5?） 5?代表这个维的长度是该批唯一最大类别长度(类别数目不足该长度的会话补0)，根据不同批会变化
        hidden = self.embedding(inputs) #返回的hidden的shape -->（100-batch_size,5?,100-embeding_size）
        hidden = self.gnn(A, hidden)
        return hidden  #(100,5?,100)


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):  #传入模型model(SessionGraph), 数据批的索引i, 训练的数据data(Data)
    #返回：动作序列对应唯一动作集合的位置角标，该批数据图矩阵的列表，单个点击动作序列的唯一类别并按照批最大类别补全0列表，面罩，目标数据
    alias_inputs, A, items, mask, targets = data.get_slice(i)  
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())  #(100,16?)
    test_alias_inputs = alias_inputs.numpy()  #测试查看alias_inputs的内容
    strange = torch.arange(len(alias_inputs)).long() #0到99
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)  #这里调用了SessionGraph的forward函数,返回维度数目(100,5?,100)
    get = lambda i: hidden[i][alias_inputs[i]]   #选择第这一批第i个样本对应类别序列的函数
    test_get = get(0)  # (16?,100)
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  #(100,16?,100)
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data): #传入模型SessionGraph，训练数据和测试数据Data
    model.scheduler.step()  #调度设置优化器的参数
    print('start training: ', datetime.datetime.now())
    model.train()  # 指定模型为训练模式，计算梯度
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):   #根据批的索引数据进行数据提取训练:i-->批索引, j-->第几批
        model.optimizer.zero_grad()  #前一步的损失清零
        targets, scores = forward(model, i, train_data) #
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward() # 反向传播
        model.optimizer.step()  # 优化
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()  # 指定模型为计算模式
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
