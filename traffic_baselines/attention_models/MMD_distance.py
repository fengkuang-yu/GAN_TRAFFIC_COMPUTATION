# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   MMD_distance.py
@Time    :   2019/10/24 16:23
@Desc    :
"""
import random

import matplotlib.pyplot as plt
# import torch
import numpy as np
import pandas as pd
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class test_demo:
    def __init__(self):
        self.cols = 60
        self.rows = 60
        self.img_shape = (self.rows, self.cols)

        self.test_percent = 0.1  # 测试集所占比例
        self.scalar = MinMaxScaler()  # 归一化的类
        self.data = self.load_data(r'D:\Users\yyh\Pycharm_workspace\GAN_TRAFFIC_COMPUTATION\traffic_data\data_all.csv')
        self.index = np.array([[x] for x in range(len(self.data))])  # 假象的标签用来表示当前取出的样本在原始文件位置
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.index, test_size=self.test_percent, random_state=12)  # 切分训练集和测试集

    def load_data(self, dir):
        # 从excel表中选出需要的数据
        traffic_data = pd.read_csv(dir, index_col=0)
        data = traffic_data.values[:, 80: 140]  # 选择数据
        time_steps = self.cols if self.cols else 1

        # 数据处理
        size = data.shape
        data = np.array(data)
        reshaped_data = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
        for i in range(size[0] - time_steps + 1):
            reshaped_data[i, :] = data[i:i + time_steps, :].flatten()

        data = self.scalar.fit_transform(reshaped_data)
        data = data.reshape(-1, *self.img_shape)
        return data


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''

    n_samples = np.int(source.shape[0] + target.shape[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    # 将source,target按列方向合并
    total = np.concatenate([source, target], axis=0)
    # 将total复制（n+m）份
    totalx = np.expand_dims(total, axis=0)
    total0 = totalx
    for i in range(int(total.shape[0]) - 1):
        total0 = np.concatenate([total0, totalx], axis=0)
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    totaly = np.expand_dims(total, axis=1)
    total1 = totaly
    for i in range(int(total.shape[0]) - 1):
        total1 = np.concatenate([total1, totaly], axis=1)
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [np.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = np.int(source.shape[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = np.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算

if __name__ == '__main__':
    demo = test_demo()
    index1 = np.random.randint(0, demo.X_train.shape[0], 100)
    index2 = np.random.randint(0, demo.X_train.shape[0], 100)
    index3 = np.random.randint(0, demo.X_train.shape[0], 100)
    images1 = demo.X_train[index1]
    images2 = np.transpose(demo.X_train[index2], [0, 2, 1])
    images3 = demo.X_train[index3]
    print(mmd_rbf(images1, images2))
    print(mmd_rbf(images1, images3))

