# -*- coding: utf-8 -*-
# @Time    : 2020/8/3 15:43
# @Author  : haowus919@gmail.com
# @File    : knn.py
# @Software: PyCharm

import numpy as np
# from sklearn.neighbors import KNeighborsClassifier as kNN
# trainmat = numpy.array([[1,2,3],[2,3,5],[55,33,66],[55,33,66]])
# label = numpy.array([0,0,1,1])
# neigh = kNN(n_neighbors=3, algorithm='auto', weights='distance', n_jobs=1)
# neigh.fit(trainmat,label)
# testmat = numpy.array([[2,3,4],[55,33,66]])
# predict = neigh.predict(testmat)
# predict_pr = neigh.predict_proba(testmat)

vector1 = np.array([1,2,3])
vector2 = np.array([4,5,6])
vector3 = np.array([2,2,3])
vector4 = np.array([4,2,7])

vector = np.array([[0.35704431, 0.58256039, 0.0, 0.18206442, 0.39341426],
                   [0.12184419, 0.53066539, 0.03994124, 0.18240998, 0.38735361],
                   [0.04630039, 0.53962414, 0.14193251, 0.18206442, 0.38205421],
                   [0.04630039, 0.53066539, 0.19945412, 0.18206442, 0.38205421],
                   [0.04630039, 0.53066539, 0.456283, 0.18206442, 0.38205421]])
len = len(vector)  #数组的长度
dict_all = {}   #key:节点   value:[dict，dict。。。。]
for i in range(0,len):
    valList=[]
    for j in range(0,len):
        dict_in = {}   #key:节点  value:两个节点的距离
        if i!=j:
            op=np.linalg.norm(vector[i]-vector[j])
            dict_in[j]=op
            valList.append(dict_in)
    dict_all[i]=valList

print(dict_all)

#
# print(op)
#输出:
#5.19615242271
#5.19615242271


