#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/8/17 15:13
# software: PyCharm

# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/7/20 9:32
# software: PyCharm

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Lambda, LeakyReLU, CuDNNGRU, GRU, TimeDistributed
from keras.layers import Conv2D, ZeroPadding2D, Flatten, BatchNormalization, Bidirectional
from keras.layers import UpSampling2D, Activation, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K

# 根据需要动态分配显存
import tensorflow as tf
import logging
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

# 配置日志
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(levelname)s %(message)s", "%X"))
logger.addHandler(handler)

# Tensorflow只打印error日志通知
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AE:
    def __init__(self, data_path):
        self.batch_size = 100
        self.cols = 60
        self.rows = 60
        self.channels = 1
        self.img_shape = (self.rows, self.cols, self.channels)
        self.latent_dim = 200
        self.intermediate_dim = 256
        self.epochs = 10000
        self.test_percent = 0.3
        self.optimizer = Adam()
        
        self.missing_percentage = 0.2
        self.scalar = MinMaxScaler()
        
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model = self.build_model()
        logger.info("************* model ready!!! ")
        self.data = self.load_data(data_path)
        logger.info("************* data ready!!! ")
    
    def build_model(self):
        real_x = Input(shape=self.img_shape)
        masks = Input(shape=self.img_shape)
        
        masked_x = Lambda(lambda x: x[0] * x[1], output_shape=self.img_shape, )([real_x, masks])
        latent = self.encoder(masked_x)
        fake_x = self.decoder(latent)
        imputed_img = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2], output_shape=self.img_shape,
                             name='imputation_layer')([masks, real_x, fake_x])
        
        ae = Model([real_x, masks], imputed_img)
        xent_loss = K.mean(K.binary_crossentropy(real_x, fake_x))
        
        ae.add_loss(xent_loss)
        ae.compile(optimizer=self.optimizer)
        ae.summary()
        return ae
    
    def build_encoder(self):
        img = Input(shape=self.img_shape)
        
        reshaped = Reshape((self.rows, self.cols))(img)
        lstm_layer1 = Bidirectional(CuDNNGRU(64, return_sequences=True))(reshaped)
        norm_layer1 = BatchNormalization()(lstm_layer1)
        lstm_layer2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(norm_layer1)
        norm_layer2 = BatchNormalization()(lstm_layer2)
        lstm_layer3 = Bidirectional(CuDNNGRU(64, return_sequences=True))(norm_layer2)
        
        fc1 = Flatten()(lstm_layer3)
        fc2 = Dense(self.intermediate_dim)(fc1)
        fc3 = LeakyReLU(alpha=0.2)(fc2)
        f_out = Dense(self.latent_dim)(fc3)
        return Model(img, f_out, name='encoder')
    
    def build_decoder(self):
        model = Sequential()
        
        model.add(Dense(32 * 15 * 15, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((15, 15, 32)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("sigmoid"))
        
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return Model(noise, img)

    def build_critic(self):
    
        img = Input(shape=self.img_shape)
        conv1 = Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(img)
        # conv1 = BatchNormalization(momentum=0.8)(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
    
        conv2 = Conv2D(64, kernel_size=3, strides=2, padding="same")(conv1)
        conv2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(conv2)  # 判断是否需要填充
        # conv2 = BatchNormalization(momentum=0.8)(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
    
        conv3 = Conv2D(128, kernel_size=3, strides=2, padding="same")(conv2)
        # conv3 = BatchNormalization(momentum=0.8)(conv3)
        conv3_output = LeakyReLU(alpha=0.2)(conv3)
    
        fc1 = Flatten()(conv3_output)
        valid = Dense(1)(fc1)
    
        return Model(img, [valid, fc1])

    def train(self):
        x_train = self.data
        logger.info("************* start training!!! ")
        for epoch in tqdm(range(self.epochs)):
            idx = np.random.randint(0, x_train.shape[0] * (1 - self.test_percent), self.batch_size)
            real_imgs = x_train[idx]
            real_imgs = real_imgs.reshape(-1, *self.img_shape)
            masks = self.patch_mask_randomly(real_imgs)
            loss = self.model.train_on_batch([real_imgs, masks], None)
            
            if epoch % 400 == 0:
                # print('loss:{}'.format(loss))
                self.plot_test('miss_percentage{}_epoch:{}.png'.format(self.missing_percentage, epoch))
    
    def mask_randomly(self, shape=None):
        if shape == None:
            random_mask = np.random.random(size=(self.batch_size, self.rows, self.cols, self.channels))
        else:
            random_mask = np.random.random(size=shape)
        random_mask[random_mask > self.missing_percentage] = 1
        random_mask[random_mask != 1] = 0
        return random_mask
    
    def patch_mask_randomly(self, images, mask_size=(12, 12)):
        """
        接收一个四维的矩阵，返回已经处理过的矩阵
        :param images:
        :param mask_height:
        :param mask_width:
        :return:
        """
        mask_height, mask_width = mask_size
        img_shape = images.shape
        img_rows = img_shape[1]
        img_width = img_shape[2]
        mask_rows_start = np.random.randint(0, img_rows - mask_height, img_shape[0])
        mask_rows_end = mask_rows_start + mask_height
        mask_cols_start = np.random.randint(0, img_width - mask_width, img_shape[0])
        mask_cols_end = mask_cols_start + mask_width
        
        masks = np.ones_like(images)
        for i, img in enumerate(images):
            _y1, _y2, _x1, _x2 = mask_rows_start[i], mask_rows_end[i], mask_cols_start[i], mask_cols_end[i],
            masks[i][_y1:_y2, _x1:_x2, :] = 0
        return masks
    
    def plot_test(self, file_name, ):
        x_test = self.data[-int(self.data.shape[0] * self.test_percent):]
        # idx = np.random.randint(0, 100, self.batch_size)
        # x_test = x_test[idx]
        masks = self.patch_mask_randomly(x_test)
        corrupted = x_test * masks
        gen_imgs = self.model.predict([x_test, masks])
        gen_imgs = gen_imgs.reshape(-1, self.rows, self.cols, self.channels)
        x_flatten = x_test.reshape((-1, self.rows * self.cols * self.channels))
        g_flatten = gen_imgs.reshape((-1, self.rows * self.cols * self.channels))
        x_flatten = self.scalar.inverse_transform(x_flatten)
        g_flatten = self.scalar.inverse_transform(g_flatten)
        mape = np.sum(np.abs(g_flatten - x_flatten) / x_flatten) / np.sum(1 - masks)
        mae = np.sum(np.abs(g_flatten - x_flatten)) / np.sum(1 - masks)
        r, c = 2, 10
        fig, axs = plt.subplots(r * 4 + 1, c)
        for j in range(c):
            for index, temp in enumerate([x_test, masks, corrupted, gen_imgs]):
                axs[index, j].imshow(temp[j, :, :, 0], cmap='gray')
                axs[index, j].axis('off')
        for j in range(c):
            axs[4, j].axis('off')
        for j in range(c):
            for index, temp in enumerate([x_test, masks, corrupted, gen_imgs]):
                axs[5 + index, j].imshow(temp[c + j, :, :, 0], cmap='gray')
                axs[5 + index, j].axis('off')
        fig.suptitle('mape:{};mae:{}'.format(mape, mae))
        fig.savefig(os.path.join(os.getcwd(), 'generated_imgs', file_name))
        plt.close()
    
    def load_data(self, file_path):
        traffic_data = pd.read_csv(file_path, index_col=0)
        data = traffic_data.values[:, 80: 140]  # 选择数据
        data = self.data_pro(data, time_steps=60)
        data = self.scalar.fit_transform(data)
        data = data.reshape(-1, *self.img_shape)
        return data
    
    def data_pro(self, data, time_steps=None):
        """
        数据处理，将列状的数据拓展开为行形式

        :param data: 输入交通数据
        :param time_steps: 分割时间长度
        :return: 处理过的数据
        """
        # 数据的预处理操作，包括缺失值填补
        data_daily = data.reshape(58, 288, 60)
        daily_average = data_daily.mean(axis=0)
        for temp in zip(*np.where(data_daily == 0)):
            data_daily[temp] = round(daily_average[temp[1:]])
        data_daily = data_daily.reshape(-1, 60)
        
        # 数据的训练和测试样本构造
        if time_steps is None:
            time_steps = 1
        size = data_daily.shape
        temp = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
        for i in range(size[0] - time_steps + 1):
            temp[i, :] = data_daily[i:i + time_steps, :].flatten()
        return temp


if __name__ == '__main__':
    img_name = r'imputated.png'
    data_path = r'traffic_data/data_all.csv'
    ae = AE(data_path)
    ae.train()
    ae.plot_test(img_name)
