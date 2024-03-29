import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
l2_reg = tf.contrib.layers.l2_regularizer(scale=5e-4)
def get_graph(path):
    """

    :param path: 邻接矩阵文件，A
    :param select_roads_num:
    :return:
    """
    # path = r'E:\workspace\PycharmProjects\untitled\GCN-GAN数据预测\参考代码\califorlia_adj.csv'
    adj = pd.read_csv(path, header=None).values
    # adj = adj[0:select_roads_num]
    degree = np.sum(adj, axis=1, dtype=np.float32)  # 邻接矩阵的度矩阵
    degree = degree + 1  # 度矩阵D，值加一
    d_inv_sqrt = np.power(degree, -0.5)  # D的（-1/2）次方
    d_inv = np.power(degree, -1)  # D的（-1）次方
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)  # 构建成对角矩阵
    d_mat_inv = np.diag(d_inv)
    graph_mean = (d_mat_inv_sqrt * adj * d_mat_inv_sqrt).astype(
        np.float32)                                                    # 公式D**(-1/2) * A * D**(-1/2)
    # graph_var = (d_mat_inv * adj * d_mat_inv).astype(np.float32)  # 公式D**(-1) * A * D**(-1)
    return graph_mean

def weight_get(name,shape):
    return tf.get_variable(name,
                    shape=shape,
                    regularizer=l2_reg,
                    initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
def bias_get(name, shape):
    return tf.get_variable(name,
                           shape=shape,
                           initializer=tf.constant_initializer(0.0, dtype=tf.float32))

def gcn_model(inputs, adj, hidden_list):
    inputs = tf.matmul(adj, inputs)
    W1 = weight_get('W1',[inputs.shape[-1], hidden_list[0]])
    W2 = weight_get('W2',[ hidden_list[0], hidden_list[1]])
    W3 = weight_get('W3',[ hidden_list[1],class_num])
    b1 = tf.placeholder(tf.float32,[class_num])
    h1= tf.nn.relu(tf.matmul(inputs,W1))
    h2 = tf.nn.relu(tf.matmul(h1,W2))
    logits = tf.matmul(h2,W3)
    outputs = tf.nn.softmax(logits)
    return logits, outputs

#数据部分
import numpy as np  # 引入numpy
from scipy import io

load_list = ["Air_Force_One", 'Base_jumping', 'Bearpark_climbing', 'Bike_Polo',
                 'Bus_in_Rock_Tunnel', 'car_over_camera', 'Car_railcrossing',
                 'Cockpit_Landing', 'Cooking', 'Eiffel_Tower', 'Excavators_river_crossing',
                 'Fire_Domino', 'Jumps', 'Kids_playing_in_leaves', 'Notre_Dame', 'Paintball',
                 'paluma_jump', 'playing_ball', 'Playing_on_water_slide', 'Saving_dolphines',
                 'Scuba', 'St_Maarten_Landing', 'Statue_of_Liberty', 'Uncut_Evening_Flight',
                 'Valparaiso_Downhill']


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


    gatherAll.append(gatherOut)   #test_x
    y_All.append(y_out)    #test_y

import networkx as nx
list = []
for i in range(0,899):
    j1 = i+1
    j2 = i+2
    c1 = i,j1
    c2 = i,j2
    list.append(c1)
    list.append(c2)

list.remove((898, 900))

G = nx.Graph(list)
A = nx.adjacency_matrix(G)
adj_A = A.todense()    #邻接矩阵



def plot_confusion_matrix(cm, savename='nn', title='Confusion Matrix'):
    classes = np.arange(class_num)
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    # plt.savefig(savename, format='png')
    plt.show()
X, Y = np.asarray(gatherAll), np.asarray(y_All)[:,:,:3]
adj = np.asarray(adj_A)
adj = tf.cast(adj, tf.float32)
train_x, train_y = X[:24,], Y[:24,]

# path = ''
# adj_path = ''

# train_x,train_y = np.zeros(shape=(100,23,34)), np.ones(shape=(100,5))

test_x, test_y = X[24:,], Y[24:,]
node_num, features,hidden_list = 900, 5, [128,64]
class_num, epoch =3, 1000
inputs = tf.placeholder(tf.float32, shape=[None, node_num, features])
# adj = get_graph(adj_path)

y_true = tf.placeholder(tf.float32, shape=[None, node_num, class_num])
logits,outputs = gcn_model(inputs, adj, hidden_list)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))

optim = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    losses = []
    for i in range(epoch):
        _, train_loss = sess.run([optim,loss],feed_dict={inputs:train_x,
                                                         y_true:train_y})
        print('epoch = {}, train_loss = {}'.format(i,train_loss))
        losses.append(train_loss)
    pres = sess.run(outputs, feed_dict={inputs:test_x})
    press = pres.reshape(-1, pres.shape[-1])
    pre_labels = np.argmax(press, axis=1)
    true_labels = np.argmax(test_y.reshape(-1,test_y.shape[-1]),axis=1)
    acc = accuracy_score(true_labels, pre_labels)
    print('Acc = {}'.format(acc))
    result = np.column_stack((pre_labels.reshape(-1,1), true_labels.reshape(-1,1)))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(losses)
    plt.show()
    cm = confusion_matrix(true_labels, pre_labels)
    plot_confusion_matrix(cm, title='Confusion Matrix')
# Set up plot



