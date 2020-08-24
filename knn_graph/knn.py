# -*- coding: utf-8 -*-
# @Time    : 2020/8/3 15:43
# @Author  : haowus919@gmail.com
# @File    : knn.py
# @Software: PyCharm
# 欧式距离 构造图网络

import numpy as np

vector = np.asarray([[0.35704431, 0.58256039, 0.0, 0.18206442, 0.39341426],
                   [0.12184419, 0.53066539, 0.03994124, 0.18240998, 0.38735361],
                   [0.04630039, 0.53962414, 0.14193251, 0.18206442, 0.38205421],
                   [0.04630039, 0.53066539, 0.19945412, 0.18206442, 0.38205421],
                   [0.04630039, 0.53066539, 0.456283, 0.18206442, 0.38205421]])
# lens = vector.shape[0]  #数组的长度

all_vals = []
for i in range(vector.shape[0]):
    vals = []
    for j in range(vector.shape[0]):
        if i!=j:
            op=np.linalg.norm(vector[i]-vector[j])
            vals.append([i,j,op])
    vals.sort(key=lambda val:val[2])
    all_vals.append(vals[:2])




