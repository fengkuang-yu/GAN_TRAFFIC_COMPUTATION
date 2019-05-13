import os

import numpy as np
import tensorflow as tf
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data

# 读入本地的MNIST数据集，该函数为mnist专用
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

batch_size = 100  # 每个batch的大小
width, height = 28, 28  # 每张图片包含28*28个像素点
mnist_dim = width * height  # 用一个数字数组表示一张图，那么这个数组展开成向量的长度就是28*28=784
random_dim = 10  # 每张图表示一个数字，从0到9
epochs = 1000000  # 共100万轮


def my_init(size):  # 从[-0.05,0.05]的均匀分布中采样得到维度是size的输出
    return tf.random_uniform(size, -0.05, 0.05)


# 判别器相关参数设定
D_W1 = tf.Variable(my_init([mnist_dim, 128]))  # 784*128
D_b1 = tf.Variable(tf.zeros([128]))  # 长度为128的一维张量，值均为0
D_W2 = tf.Variable(my_init([128, 32]))
D_b2 = tf.Variable(tf.zeros([32]))
D_W3 = tf.Variable(my_init([32, 1]))
D_b3 = tf.Variable(tf.zeros([1]))
D_variables = [D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]

# 生成器相关参数设定
G_W1 = tf.Variable(my_init([random_dim, 32]))
G_b1 = tf.Variable(tf.zeros([32]))
G_W2 = tf.Variable(my_init([32, 128]))
G_b2 = tf.Variable(tf.zeros([128]))
G_W3 = tf.Variable(my_init([128, mnist_dim]))
G_b3 = tf.Variable(tf.zeros([mnist_dim]))
G_variables = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]


# 判别器网络结构
def D(X):
    X = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)  # X的维度是100*784，D_W1维度是784*128，得到结果维度为100*128
    X = tf.nn.relu(tf.matmul(X, D_W2) + D_b2)  # X的维度是100*128，D_W2维度是128*32，得到结果维度为100*32
    X = tf.matmul(X, D_W3) + D_b3  # X的维度是100*32，D_W3维度是32*1，得到结果维度为100*1
    return X


# 生成器网络结构
def G(X):
    X = tf.nn.relu(tf.matmul(X, G_W1) + G_b1)  # X的维度是100*10，G_W1维度是10*32，得到结果维度为100*32
    X = tf.nn.relu(tf.matmul(X, G_W2) + G_b2)  # X的维度是100*32，G_W2维度是32*128，得到结果维度为100*128
    X = tf.nn.sigmoid(tf.matmul(X, G_W3) + G_b3)  # X的维度是100*128，G_W3维度是128*784，得到结果维度为100*784
    return X


# real_X是真实样本，random_X是噪音数据，random_Y是生成器生成的伪样本
real_X = tf.placeholder(tf.float32, shape=[batch_size, mnist_dim])
random_X = tf.placeholder(tf.float32, shape=[batch_size, random_dim])
random_Y = G(random_X)

# 求惩罚项，这个这个惩罚是“软约束”，最终的结果不一定满足这个约束，但是会在约束上下波动。这里Lipschitz约束的C=1
eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)  # eps是U[0,1]的随机数
X_inter = eps * real_X + (1. - eps) * random_Y  # 在真实样本和生成样本之间随机插值，希望这个约束可以“布满”真实样本和生成样本之间的空间
grad = tf.gradients(D(X_inter), [X_inter])[0]  # 求梯度
grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2, axis=1))  # 求梯度的二范数
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))  # Lipschitz限制是要求判别器的梯度不超过K，这个loss项是希望判别器的梯度离K（此处K设为1）越近越好

# 判别器和生成器的损失函数
D_loss = tf.reduce_mean(D(real_X)) - tf.reduce_mean(D(random_Y)) + grad_pen
G_loss = tf.reduce_mean(D(random_Y))  # 越接近真实样本越好

# 判别器和生成器的优化函数
D_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(D_loss, var_list=D_variables)
G_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(G_loss, var_list=G_variables)

# 创建对话，初始化所有变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 是否存在“out”文件夹，不存在的话新建一个，存放实验结果
if not os.path.exists('out/'):
    os.makedirs('out/')

for e in range(epochs):
    for i in range(5):  # 每轮计算5个batch
        real_batch_X, _ = mnist.train.next_batch(batch_size)  # 随机抓取训练数据中的100个批处理数据点
        random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim))  # 从均匀分布中采样，输出100*10个样本
        _, D_loss_ = sess.run([D_solver, D_loss], feed_dict={real_X: real_batch_X, random_X: random_batch_X})
    random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim))
    _, G_loss_ = sess.run([G_solver, G_loss], feed_dict={random_X: random_batch_X})
    # 每1000轮输出一次当前结果
    if e % 1000 == 0:
        print('epoch %s, D_loss: %s, G_loss: %s' % (e, D_loss_, G_loss_))
        n_rows = 6
        check_imgs = sess.run(random_Y, feed_dict={random_X: random_batch_X}).reshape((batch_size, width, height))[
                     :n_rows * n_rows]  # 由生成器得到伪样本，维度为100*784，reshape为100个28*28的矩阵，取6*6个矩阵构成一张图
        imgs = np.ones((width * n_rows + 5 * n_rows + 5, height * n_rows + 5 * n_rows + 5))  # 203*203的值为1的二维矩阵
        for i in range(n_rows * n_rows):
            imgs[5 + 5 * (i % n_rows) + width * (i % n_rows):5 + 5 * (i % n_rows) + width + width * (i % n_rows),
            5 + 5 * (i / n_rows) + height * (i / n_rows):5 + 5 * (i / n_rows) + height + height * (i / n_rows)] = \
            check_imgs[i]
        misc.imsave('out/%s.png' % (e / 1000), imgs)

































