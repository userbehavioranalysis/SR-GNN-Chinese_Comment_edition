# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 22:18:19 2019

@author: dell
"""

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)


#TEST_X = [[3,7,7,3,6],[1,2,3,2,4] ]
#TEST_Y = [4,5]
#
#for sequences, y in zip(TEST_X, TEST_Y):
#    i = 0
#    nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
#    senders = []
#    x = []
#    for node in sequences:
#        if node not in nodes:
#            nodes[node] = i
#            x.append([node])
#            i += 1
#        senders.append(nodes[node])
#    receivers = senders[:]
#    del senders[-1]    # the last item is a receiver
#    del receivers[0]    # the first item is a sender
    
