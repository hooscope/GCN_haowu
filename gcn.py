import tensorflow as tf
import pandas as pd
import numpy as np
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
                    initializer=tf.contrib.layers.xavier_initializer())
def bias_get(name, shape):
    return tf.get_variable(name,
                           shape=shape,
                           initializer=tf.constant_initializer(0.0, dtype=tf.float32))

def gcn_model(inputs, adj, hidden_list):
    inputs = tf.matmul(adj, inputs)
    W1 = weight_get('W1',[inputs.shape[-1], hidden_list[0]])
    W2 = weight_get('W2',[ hidden_list[0], hidden_list[1]])
    W3 = weight_get('W3',[ hidden_list[1], 1])
    b1 = tf.placeholder(tf.float32,[1])
    h1= tf.nn.relu(tf.matmul(inputs,W1))
    h2 = tf.nn.relu(tf.matmul(h1,W2))
    logits = tf.matmul(h2,W3) + b1
    outputs = tf.nn.softmax(logits)
    return h2, outputs



path = ''
adj_path = ''

train_x,train_y = np.zeros(shape=(100,23,34)), np.ones(shape=(100,5))

node_num, features,hidden_list = 4000, 5, [128,64]
class_num, epoch = 5, 100
inputs = tf.placeholder(tf.float32, shape=[None, node_num, features])
y_true = tf.placeholder(tf.float32, shape=[None, class_num])
adj = get_graph(adj_path)

logits,outputs = gcn_model(inputs, adj, hidden_list)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in range(epoch):
        _, train_loss = sess.run([optim,loss],feed_dict={inputs:train_x,
                                                         y_true:train_y})




    pres = sess.run(outputs, feed_dict={inputs:test_x})
