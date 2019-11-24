#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/11/22 11:22
# software: PyCharm
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/10/16 15:30
# software: PyCharm

import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

from sklearn
from keras.optimizers import Adam
from keras.utils import plot_model
from keras_multi_head.multi_head_attention import MultiHeadAttention
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils import FeedForward
from utils import LayerNormalization
from utils import PositionEmbedding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
K.set_session(session)


class KNN_imputation():
    def __init__(self):
        # 设置训练超参数
        self.batch_size = 100
        self.cols = 60
        self.rows = 60
        self.img_shape = (self.rows, self.cols)
        self.multi_head_num = 6
        self.ff_hidden_unit_num = 1024
        self.epochs = 10000
        self.optimizer = Adam(beta_1=0.9, beta_2=0.98, epsilon=10e-9)
        self.loss_list = []
        
        # 构建模型
        self.attention_model = self.build_attention_model_with_single_input()
        
        # 数据处理相关
        self.missing_percentage = 0.3
        self.test_percent = 0.1
        self.scalar = MinMaxScaler()
        self.data = self.load_data(r'traffic_data/data_all.csv')
        self.index = np.array([[x] for x in range(len(self.data))])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.index, test_size=self.test_percent, random_state=12)
        self.miss_mode = 'patch'
    
    def build_attention_model_with_single_input(self):
    
    
    def plot_test(self, file_name=None):
        
        img_index = np.random.randint(0, self.X_test.shape[0], 100)
        x_test = self.X_test[img_index].reshape(-1, *self.img_shape)
        x_gen = self.attention_model.predict(x_test)
        
        r, c = 4, 4
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(x_gen[cnt, :, :], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.subplots_adjust(wspace=0.9, hspace=0.9)
        fig.suptitle('Generated images')
        # fig.savefig(os.path.join(os.getcwd(), 'generated_imgs', file_name))
        plt.show()
        plt.close()
        
        r, c = 4, 4
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(x_test[cnt, :, :], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.subplots_adjust(wspace=0.9, hspace=0.9)
        fig.suptitle('Real images')
        # fig.savefig(os.path.join(os.getcwd(), 'generated_imgs', file_name))
        plt.show()
        plt.close()
    
    def mask_randomly(self, shape, mode='patch'):
        """
        接收一个三维以上的矩阵，返回已经处理过的矩阵
        :param images:
        :param mode:
        :param percent:
        :param mask_height:
        :param mask_width:
        :return:
        """
        assert len(shape) >= 3
        assert isinstance(shape, tuple)
        percent = self.missing_percentage if self.missing_percentage else 0.1
        masks = np.ones(shape=shape)
        if mode == 'patch':
            img_rows, img_width = shape[1], shape[2]
            mask_height = mask_width = int(np.sqrt(img_rows * img_width * percent))
            
            assert img_rows - mask_height > 0
            assert img_width - mask_width > 0
            mask_rows_start = np.random.randint(0, img_rows - mask_height, shape[0])
            mask_rows_end = mask_rows_start + mask_height
            mask_cols_start = np.random.randint(0, img_width - mask_width, shape[0])
            mask_cols_end = mask_cols_start + mask_width
            
            for i in range(shape[0]):
                _y1, _y2, _x1, _x2 = mask_rows_start[i], mask_rows_end[i], mask_cols_start[i], mask_cols_end[i],
                masks[i][_y1:_y2, _x1:_x2] = 0
            return masks
        
        elif mode == 'spatial_line':
            for i in range(shape[0]):
                index_cols = np.random.randint(0, shape[2], int(percent * shape[2]))
                masks[i][:, index_cols] = 0
            return masks
        
        elif mode == 'temporal_line':
            for i in range(shape[0]):
                index_rows = np.random.randint(0, shape[1], int(percent * shape[1]))
                masks[i][index_rows, :] = 0
            return masks
        
        elif mode == 'random':
            pass
    
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
        
        # data = self.data_pro(data, time_steps=60)
        data = self.scalar.fit_transform(reshaped_data)
        data = data.reshape(-1, *self.img_shape)
        return data
    
    def plot_test_mask(self, file_name, full=False):
        x_test = self.X_test
        idx = np.random.randint(0, 100, self.batch_size)
        x_test = x_test[idx] if not full else x_test
        masks = self.mask_randomly(x_test.shape, mode=self.miss_mode)
        corrupted = x_test * masks
        gen_imgs, masks_gen = self.attention_model.predict([x_test, masks], batch_size=self.batch_size)
        x_flatten = x_test.reshape((-1, self.rows * self.cols))
        g_flatten = gen_imgs.reshape((-1, self.rows * self.cols))
        m_flatten = masks_gen.reshape((-1, self.rows * self.cols))
        x_flatten = self.scalar.inverse_transform(x_flatten)
        g_flatten = self.scalar.inverse_transform(g_flatten)
        m_flatten = self.scalar.inverse_transform(m_flatten)
        
        # 生成样本的误差
        total_rmse = np.sqrt(np.sum(np.power(g_flatten - x_flatten, 2)) / np.prod(x_flatten.shape))
        total_mae = np.sum(np.abs(g_flatten - x_flatten)) / np.prod(x_flatten.shape)
        
        # 填充部分的误差
        masks_rmse = np.sqrt(np.sum(np.power(m_flatten - x_flatten, 2)) / np.sum(1 - masks))
        masks_mae = np.sum(np.abs(m_flatten - x_flatten)) / np.sum(1 - masks)
        
        r, c = 2, 8
        fig, axs = plt.subplots(r * 3 + 1, c)
        for j in range(c):
            for index, temp in enumerate([x_test, corrupted, gen_imgs]):
                axs[index, j].imshow(temp[j, :, :], cmap='gray')
                axs[index, j].axis('off')
        for j in range(c):
            axs[3, j].axis('off')
        for j in range(c):
            for index, temp in enumerate([x_test, corrupted, gen_imgs]):
                axs[4 + index, j].imshow(temp[c + j, :, :], cmap='gray')
                axs[4 + index, j].axis('off')
        fig.suptitle('total_rmse:{:.3f};total_mae:{:.3f}\n'
                     'masks_rmse:{:.3f};masks_mae:{:.3f}'.format(total_rmse, total_mae, masks_rmse, masks_mae))
        fig.savefig(os.path.join(os.getcwd(), 'generated_images', 'ae', file_name), dpi=300)
        plt.close()
    
    def train(self, epochs=20000):
        # x_train = self.x_train
        x_train = self.X_train
        total_epoch = self.epochs if epochs == None else epochs
        
        for epoch in tqdm(range(total_epoch)):
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_images = x_train[idx]
            real_images = real_images.reshape((-1, *self.img_shape))
            masks = self.mask_randomly(real_images.shape, mode=self.miss_mode)
            loss = self.attention_model.train_on_batch([real_images, masks], None)
            
            if epoch % 10 == 0:
                # self.plot_test_mask(
                #     self.miss_mode + '{:.1%}_{:0>5d}epochs_gen.png'.format(self.missing_percentage, epoch)
                # )
                # print('loss:{}'.format(loss))
                self.loss_list.append((epoch, loss))
    
    def plot_loss(self):
        loss = self.loss_list
        loss = pd.DataFrame(data=[x[1] for x in loss], index=[x[0] for x in loss], columns=['reconstruct_loss'])
        
        loss.plot()
        plt.savefig(os.path.join(os.getcwd(), 'training_related_img', 'autoencoder_loss.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    iterations = 20000
    ae = KNN_imputation()
    ae.train(iterations)
    ae.plot_test_mask(ae.miss_mode + '{:.1%}_{:0>5d}epochs_gen.png'.format(ae.missing_percentage, iterations),
                      full=True)
    ae.plot_loss()
