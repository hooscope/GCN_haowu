# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 15:19
# @Author  : haowus919@gmail.com
# @File    : adjMatrix.py
# @Software: PyCharm

import scipy as sp
import networkx as nx

# lst = [('zs','ww'),('zs','ll'),('zs','wq'),('ww','zs')]
# lst = [(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4),(3,5)]
# G = nx.Graph(lst)
# A = nx.adjacency_matrix(G)
# print(A.todense())


list = []
for i in range(0,899):
    j1 = i+1
    j2 = i+2
    c1 = i,j1
    c2 = i,j2
    list.append(c1)
    list.append(c2)

list.remove((898, 900))
print(list)

G = nx.Graph(list)
A = nx.adjacency_matrix(G)
adj_A = A.todense()
print(A.todense())