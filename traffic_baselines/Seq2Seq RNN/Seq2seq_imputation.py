#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/11/30 17:32
# software: PyCharm

import os
import logging

from multiprocessing import Pool
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Add, Input, Lambda, Activation, Dense, Flatten, Reshape
from keras.layers import Bidirectional, CuDNNLSTM, TimeDistributed
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='seq2seq_model_result.log',
                    level=logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
K.set_session(session)


class seq2seq_models():
    def __init__(self, epochs=10000, missing_percentage=0.8, test_percent=0.15,
                 adam_beta_1=0.9, adam_beta_2=0.98, adam_epsilon=10e-9,
                 data_path=r'traffic_data/data_all.csv', miss_mode='patch'):
        # 设置训练超参数
        self.batch_size = 128
        self.cols = 60
        self.rows = 60
        self.img_shape = (self.rows, self.cols)
        self.ff_hidden_unit_num = 128
        self.epochs = epochs
        self.optimizer = Adam(beta_1=adam_beta_1, beta_2=adam_beta_2, epsilon=adam_epsilon)
        
        # 数据处理相关
        self.missing_percentage = missing_percentage
        self.test_percent = test_percent
        self.scalar = MinMaxScaler()
        self.data = self.load_data(data_path)
        self.index = np.array([[x % 288] for x in range(len(self.data))])  # 将标签作为输入矩阵的label，取值为时间戳
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.index, test_size=self.test_percent, random_state=12)
        self.miss_mode = miss_mode
        
        # 构建模型
        self.attention_model = self.seq2seq_model()
    
    def seq2seq_model(self):
        
        input_x = Input(shape=self.img_shape, name='real_inputs')
        masks = Input(shape=self.img_shape, name='mask_inputs')
        time_stamp = Input(shape=(1,), name='condition_input_time_stamp')
        masked_x = Lambda(lambda x: x[0] * x[1] + -1. * (1 - x[1]), output_shape=self.img_shape, name='masked_input')(
            [input_x, masks])
        
        # hidden_layer_1 = Bidirectional(CuDNNLSTM(self.ff_hidden_unit_num, return_sequences=True))(masked_x)
        # hidden_layer_2 = Bidirectional(CuDNNLSTM(self.ff_hidden_unit_num, return_sequences=True))(hidden_layer_1)
        hidden_layer = Bidirectional(CuDNNLSTM(self.ff_hidden_unit_num, return_sequences=True))(masked_x)
        fake_res = TimeDistributed(Dense(self.cols, activation='sigmoid'))(hidden_layer)
        
        imputed_img = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2], output_shape=self.img_shape,
                             name='imputation_layer')([masks, input_x, fake_res])
        
        attention_model = Model([input_x, masks, time_stamp], [fake_res, imputed_img], name='basic_generator')
        att_loss = K.mean(mean_squared_error(input_x, fake_res))
        attention_model.add_loss(att_loss)
        attention_model.compile(optimizer=self.optimizer, )
        attention_model.summary()
        # plot_model(attention_model, to_file=os.path.join(os.getcwd(), 'network_related_img', 'attention_model.pdf'))
        
        return attention_model
    
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
        test_index = self.y_test
        idx = np.random.randint(0, 100, self.batch_size)
        x_test = x_test[idx] if not full else x_test
        index = test_index[idx] if not full else test_index
        
        masks = self.mask_randomly(x_test.shape, mode=self.miss_mode)
        corrupted = x_test * masks
        
        gen_imgs, masks_gen = self.attention_model.predict([x_test, masks, index], batch_size=self.batch_size)
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
        logging.error('\n miss_mode: {} '
                     '\n miss_rate: {} '
                     '\n masks_rmse: {} '
                     '\n masks_mae: {}'.format(self.miss_mode, self.missing_percentage, masks_rmse, masks_mae))

    
    def train(self, epochs=20000):
        # x_train = self.x_train
        x_train = self.X_train
        train_index = self.y_train
        total_epoch = self.epochs if epochs == None else epochs
        
        for epoch in range(total_epoch):
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_images = x_train[idx]
            real_index = train_index[idx]
            real_images = real_images.reshape((-1, *self.img_shape))
            masks = self.mask_randomly(real_images.shape, mode=self.miss_mode)
            loss = self.attention_model.train_on_batch([real_images, masks, real_index], None)


if __name__ == '__main__':
    def base_func(miss_percent, miss_mode, iterations):
        model = seq2seq_models(miss_mode=miss_mode, missing_percentage=miss_percent)
        model.train(iterations)
        model.plot_test_mask(model.miss_mode + '{:.1%}_{:0>5d}epochs_seq2seq_lstm.png'.format(model.missing_percentage, iterations), full=True)
    
    iterations = 40000
    
    # 多进程调用
    logging.error("*"*100)
    for miss_mode in ['patch']:
        for miss_percent in [0.1 * x for x in range(1, 10)]:
            base_func(miss_percent, miss_mode, iterations)
