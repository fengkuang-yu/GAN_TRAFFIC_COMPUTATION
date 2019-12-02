#!/usr/bin/env python
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
from keras.layers import Add, Input, Lambda, Activation, Dense, Flatten, Reshape
from keras.layers import Bidirectional, LSTM
from keras.losses import mean_squared_error
from keras.models import Model
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
K.set_session(session)


class autoencoder_models():
    def __init__(self, epochs=10000, missing_percentage=0.8, test_percent=0.15,
                 adam_beta_1=0.9, adam_beta_2=0.98, adam_epsilon=10e-9,
                 data_path=r'traffic_data/data_all.csv', miss_mode='patch'):
        # 设置训练超参数
        self.batch_size = 128
        self.cols = 60
        self.rows = 60
        self.img_shape = (self.rows, self.cols)
        self.multi_head_num = 6
        self.ff_hidden_unit_num = 1024
        self.epochs = epochs
        self.optimizer = Adam(beta_1=adam_beta_1, beta_2=adam_beta_2, epsilon=adam_epsilon)
        self.loss_list = []
        
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
        self.attention_model = self.attention_model()
    
    def attention_model(self):
        
        def attention_block(x, head_num, feed_forward_hidden_units, kernel_constraint=None):
            """
            multi-head attention block
            :param input_x:
            :return:
            """
            attention_multi_head = MultiHeadAttention(head_num=head_num, kernel_constraint=kernel_constraint)(x)
            add_attention = Add()([x, attention_multi_head])
            layer_normal = LayerNormalization(gamma_constraint=kernel_constraint)(add_attention)
            feed_forward = FeedForward(units=feed_forward_hidden_units, kernel_constraint=kernel_constraint)(
                layer_normal)
            feed_forward_add = Add()([feed_forward, layer_normal])
            block_res = LayerNormalization(gamma_constraint=kernel_constraint)(feed_forward_add)
            return block_res
        
        input_x = Input(shape=self.img_shape, name='real_inputs')
        masks = Input(shape=self.img_shape, name='mask_inputs')
        time_stamp = Input(shape=(1,), name='condition_input_time_stamp')
        masked_x = Lambda(lambda x: x[0] * x[1], output_shape=self.img_shape, name='masked_input')([input_x, masks])
        
        # 以不同的空间点作为交通流量输入
        pos_encoding_1 = PositionEmbedding(start_index=time_stamp)(masked_x)
        spatial_res_1 = attention_block(x=pos_encoding_1, head_num=self.multi_head_num,
                                        feed_forward_hidden_units=self.ff_hidden_unit_num)
        
        # 以不同的时间点的交通流量作为输入
        input_masked_trans = Lambda(function=lambda x: K.permute_dimensions(x, [0, 2, 1]),
                                    output_shape=self.img_shape[::-1],
                                    name='transpose_layer_1')(masked_x)  # 将输入旋转了一下
        pos_encoding_trans = PositionEmbedding(start_index=time_stamp)(input_masked_trans)
        temporal_res_trans_1 = attention_block(x=pos_encoding_trans, head_num=self.multi_head_num,
                                               feed_forward_hidden_units=self.ff_hidden_unit_num)
        temporal_res_1 = Lambda(function=lambda x: K.permute_dimensions(x, [0, 2, 1]),
                                output_shape=self.img_shape[::-1],
                                name='transpose_layer_2')(temporal_res_trans_1)  # 将输入转置过来
        
        # 将两个特征图加起来作为后续的网络输入
        feed_forward_res_hybrid = Add()([spatial_res_1, temporal_res_1])
        
        feed_forward_res_2 = attention_block(x=feed_forward_res_hybrid, head_num=self.multi_head_num,
                                             feed_forward_hidden_units=self.ff_hidden_unit_num)
        
        feed_forward_res_3 = attention_block(x=feed_forward_res_2, head_num=self.multi_head_num,
                                             feed_forward_hidden_units=self.ff_hidden_unit_num)
        
        fake_res = Activation('sigmoid')(feed_forward_res_3)
        
        imputed_img = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2], output_shape=self.img_shape,
                             name='imputation_layer')([masks, input_x, fake_res])
        
        attention_model = Model([input_x, masks, time_stamp], [fake_res, imputed_img], name='basic_generator')
        att_loss = K.mean(mean_squared_error(input_x, fake_res))
        attention_model.add_loss(att_loss)
        attention_model.compile(optimizer=self.optimizer, )
        attention_model.summary()
        # plot_model(attention_model, to_file=os.path.join(os.getcwd(), 'network_related_img', 'attention_model.pdf'))
        
        return attention_model
    
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


class dense_ae():
    def __init__(self, epochs=10000, missing_percentage=0.8, test_percent=0.15,
                 adam_beta_1=0.9, adam_beta_2=0.98, adam_epsilon=10e-9,
                 data_path=r'traffic_data/data_all.csv', miss_mode='patch'):
        # 设置训练超参数
        self.batch_size = 100
        self.cols = 60
        self.rows = 60
        self.img_shape = (self.rows, self.cols)
        self.multi_head_num = 6
        self.ff_hidden_unit_num = 1024
        self.epochs = epochs
        self.optimizer = Adam(beta_1=adam_beta_1, beta_2=adam_beta_2, epsilon=adam_epsilon)
        self.loss_list = []
        
        # 数据处理相关
        self.missing_percentage = missing_percentage
        self.test_percent = test_percent
        self.scalar = MinMaxScaler()
        self.data = self.load_data(data_path)
        self.index = np.array([[x] for x in range(len(self.data))])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.index, test_size=self.test_percent, random_state=12)
        self.miss_mode = miss_mode
        
        # 构建模型
        self.attention_model = self.dense_model()
    
    def dense_model(self):
        input_x = Input(shape=self.img_shape, name='real_data')
        masks = Input(shape=self.img_shape, name='mask_layer')
        masked_x = Lambda(function=lambda x: x[0] * x[1] + (1 - x[1]) * -1.,
                          output_shape=self.img_shape,
                          name='masked_input')([input_x, masks])
        
        # dense_net
        input_layer = Flatten()(masked_x)
        dense_1 = Dense(128, activation='relu')(input_layer)
        dense_2 = Dense(128, activation='relu')(dense_1)
        dense_3 = Dense(np.prod(self.img_shape), activation='sigmoid')(dense_2)
        
        fake_x = Reshape(self.img_shape)(dense_3)
        
        imputed_img = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2],
                             output_shape=self.img_shape,
                             name='imputation_layer')([masks, input_x, fake_x])
        
        attention_model = Model([input_x, masks], [fake_x, imputed_img])
        
        att_loss = K.mean(mean_squared_error(input_x, fake_x))
        attention_model.add_loss(att_loss)
        attention_model.compile(optimizer=self.optimizer, )
        attention_model.summary()
        return attention_model
    
    def bi_direction_rnn_model(self):
        input_x = Input(shape=self.img_shape, name='real_data')
        masks = Input(shape=self.img_shape, name='mask_layer')
        masked_x = Lambda(function=lambda x: x[0] * x[1] + (1 - x[1]) * -1.,
                          output_shape=self.img_shape,
                          name='masked_input')([input_x, masks])
        
        layer_1 = Bidirectional(LSTM(self.ff_hidden_unit_num, return_sequences=True))(masked_x)
        layer_hidden = Bidirectional(LSTM(self.ff_hidden_unit_num, return_sequences=True))(layer_1)
        fake_x = Bidirectional(LSTM(self.ff_hidden_unit_num, return_sequences=True))(layer_hidden)
        
        imputed_img = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2],
                             output_shape=self.img_shape,
                             name='imputation_layer')([masks, input_x, fake_x])
        
        rnn_model = Model([input_x, masks], [fake_x, imputed_img])
        
        att_loss = K.mean(mean_squared_error(input_x, fake_x))
        rnn_model.add_loss(att_loss)
        rnn_model.compile(optimizer=self.optimizer, )
        rnn_model.summary()
        return rnn_model
    
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
            
    def plot_loss(self):
        loss = self.loss_list
        loss = pd.DataFrame(data=[x[1] for x in loss], index=[x[0] for x in loss], columns=['reconstruct_loss'])
        
        loss.plot()
        plt.savefig(os.path.join(os.getcwd(), 'training_related_img', 'autoencoder_loss.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    def base_func(miss_percent, miss_mode, iterations):
        ae = dense_ae(miss_mode=miss_mode, missing_percentage=miss_percent)
        ae.train(iterations)
        ae.plot_test_mask(
            ae.miss_mode + '{:.1%}_{:0>5d}epochs_dense_ae_index.png'.format(ae.missing_percentage, iterations),
            full=True)
        ae.plot_loss()
    
    
    iterations = 40000
    
    for miss_mode in ['patch', 'spatial_line', 'temporal_line']:
        for miss_percent in [0.1 * x for x in range(1, 10)]:
            base_func(miss_percent, miss_mode, iterations)
