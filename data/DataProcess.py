# -*- coding: utf-8 -*-
# @Time    : 2020/7/28 15:13
# @Author  : haowus919@gmail.com
# @File    : DataProcess.py
# @Software: PyCharm

import numpy as np  # 引入numpy
from scipy import io

load_list = ["Air_Force_One", 'Base_jumping', 'Bearpark_climbing', 'Bike_Polo',
                 'Bus_in_Rock_Tunnel', 'car_over_camera', 'Car_railcrossing',
                 'Cockpit_Landing', 'Cooking', 'Eiffel_Tower', 'Excavators_river_crossing',
                 'Fire_Domino', 'Jumps', 'Kids_playing_in_leaves', 'Notre_Dame', 'Paintball',
                 'paluma_jump', 'playing_ball', 'Playing_on_water_slide', 'Saving_dolphines',
                 'Scuba', 'St_Maarten_Landing', 'Statue_of_Liberty', 'Uncut_Evening_Flight',
                 'Valparaiso_Downhill']
a_motion = []
a_quality = []
a_aesthetics = []
a_memory = []
a_vvsc = []
y_data = []
gatherAll = []    #(25,900,5)
y_All = []  #(25,900,1)
for load_path in load_list:
    ######下载标注分数########
    # mat = io.loadmat('F:/论文/视频摘要/SumMe/GT/'+load_path+'.mat')
    mat = io.loadmat('data_xu/五个特征/专家评分/' + load_path + '.mat')
    score_gt = np.array(mat['gt_score'])  # 键值 是numpy格式

    score_motion = np.loadtxt('data_xu/五个特征/motion/' + load_path + '.txt')  # 无毛刺的版本
    score_motion = list(score_motion)

    score_quality = np.loadtxt('data_xu/五个特征/quality/' + load_path + '.txt')
    score_quality = list(score_quality)

    score_aesthetics = np.loadtxt('data_xu/五个特征/aesthetics/' + load_path + '.txt')  # snap 得分
    score_aesthetics = list(score_aesthetics)

    score_memory = np.loadtxt('data_xu/五个特征/memory/' + load_path + '.txt')
    score_memory = list(score_memory)

    score_vvsc = np.loadtxt('data_xu/五个特征/vvsc/' + load_path + '.txt')  # 分割线对应的列表coco分数
    score_vvsc = list(score_vvsc)

    # m = [x for x in
    #      range(0, min(len(score_motion), len(score_quality), len(score_aesthetics), len(score_memory), len(score_vvsc)))]

    gatherOut = []
    y_out = []
    for i in range(0,900):
        # a_motion.append(score_motion[i])
        # a_quality.append(score_quality[i])
        # a_aesthetics.append(score_aesthetics[i])
        # a_memory.append(score_memory[i])
        # a_vvsc.append(score_vvsc[i])
        gather = []
        y_gather = []
        gather.append(score_motion[i])
        gather.append(score_quality[i])
        gather.append(score_aesthetics[i])
        gather.append(score_memory[i])
        gather.append(score_vvsc[i])

        gatherOut.append(gather)
        # y_data.append(score_gt[i][0])  # 真实值
        test_y = score_gt[i][0]
        if 0<=test_y<0.2:
            test_y = [1,0,0,0,0]
        elif 0.2<=test_y<0.4:
            test_y =[0,1,0,0,0]
        elif 0.4 <= test_y < 0.6:
            test_y = [0,0,1,0,0]
        elif 0.6<=test_y<0.8:
            test_y =[0,0,0,1,0]
        elif 0.8<=test_y<=1:
            test_y =[0,0,0,0,1]
        # y_gather.append(test_y)
        y_out.append(test_y)


    gatherAll.append(gatherOut)
    y_All.append(y_out)
a_1 = np.array(a_motion)
a_2 = np.array(a_quality)
a_3 = np.array(a_aesthetics)
a_4 = np.array(a_memory)
a_5 = np.array(a_vvsc)
y = np.array(y_data)