#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/11/6 9:26
# software: PyCharm

# -*- coding: utf-8 -*-

"""
本程序的修改点主要在以下：
1. 使用缺失部分计算误差而不是使用全部的图像重构误差
2. 对缺失部分使用负数标记如-1
3. 将输入理解为两个时间序列，第一个是检测器之间的注意力，第二个是时间上的注意力
4. 减少判别器的判别能力
5. 添加时间和空间的两种交通表示
6. 添加重构误差为l1损失
7. 添加u-net结构

"""

import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K
from keras.layers import Add, Flatten, Dense, Activation, Conv2D
from keras.layers import Input, LeakyReLU, Lambda, Reshape, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
# from keras_multi_head import MultiHeadAttention
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class AttentionGAN():
    def __init__(self, train_data_path):
        # 设置超参数
        self.batch_size = 32
        self.generator_optimizer = Adam(lr=1e-3, beta_1=0., beta_2=0.9)
        self.critic_optimizer = Adam(lr=4e-3, beta_1=0., beta_2=0.9)

        # 数据处理相关
        self.test_percent = 0.15  # 测试集所占比例
        self.data = self.load_data()
        self.index = np.array([[x] for x in range(len(self.data))])  # 假象的标签用来表示当前取出的样本在原始文件位置
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.index, test_size=self.test_percent, random_state=12)  # 切分训练集和测试集

        # 构建模型
        self.generator = self.build_generator()  # basic生成器
        self.critic = self.build_critic()  # basic判别器

        self.generator_trainer = self.build_generator_trainer()
        self.critic_trainer = self.build_critic_trainer()

    def load_data(self):
        return np.random.random(size=(1000, 10))

    def build_generator(self):
        input_x = Input(shape=(10,), name='input_x')
        masks_x = Input(shape=(10,), name='masks_x')
        index = Input(shape=(1,), name='index')

        input_masked_x_minus = Lambda(function=lambda x: x[0] * x[1] + x[2], output_shape=(10,), name='masked_input')([input_x, masks_x, index])
        hidden_layer = Dense(10)(input_masked_x_minus)
        fake_res = Activation('sigmoid')(hidden_layer)

        attention_model = Model([input_x, masks_x, index], [fake_res], name='basic_generator')
        return attention_model

    def build_critic(self):
        input_x = Input(shape=(10,))
        output = Dense(10)(input_x)

        basic_critic = Model(input_x, output, name='basic_discriminator')
        return basic_critic

    def build_critic_trainer(self):
        ### build discriminator ###
        self.generator.trainable = False
        self.critic.trainable = True
        x_real = Input(shape=(10,), name='real_image')
        masks = Input(shape=(10,), name='mask_layer')
        index = Input(shape=(1,), name='index')

        x_fake = self.generator([x_real, masks, index])  # todo: 修复后的图片作为损失计算目标

        rec_valid = self.critic(x_fake)
        real_valid = self.critic(x_real)

        critic_loss = Lambda(function=lambda x: (x[0] - x[1]) ** 2, output_shape=(1,), name='critic_loss')([rec_valid, real_valid])

        critic_trainer = Model([x_real, masks, index], [critic_loss], name='discriminator')
        critic_trainer.add_loss(critic_loss)
        critic_trainer.compile(optimizer=self.critic_optimizer)
        critic_trainer.summary()
        return critic_trainer

    def build_generator_trainer(self):
        ### build generator ###
        self.generator.trainable = True
        self.critic.trainable = False
        real_img = Input(shape=(10,), name='real_image')
        masks_generator = Input(shape=(10,), name='masks')
        index = Input(shape=(1,), name='index')

        fake_img = self.generator([real_img, masks_generator, index])  # 返回两个值，第一个是纯生成的图，第二个是修复图
        real_valid_critic = self.critic(real_img)
        fake_valid_critic = self.critic(fake_img)

        gen_loss = Lambda(function=lambda x: (x[0] - x[1]) ** 2, name='generative_loss')([real_valid_critic, fake_valid_critic])

        generator_trainer = Model([real_img, masks_generator, index], [gen_loss], name='generator_trainer')
        generator_trainer.add_loss(gen_loss)
        generator_trainer.compile(optimizer=self.generator_optimizer)
        generator_trainer.summary()
        return generator_trainer

    def train(self, epochs=None):
        x_train = self.X_train
        train_index = self.y_train
        total_epoch = 100

        for epoch in tqdm(range(total_epoch)):
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_images = x_train[idx]
            real_index = train_index[idx]

            critic_loss = self.critic_trainer.train_on_batch([real_images, real_images, real_index], None)
            # 生成器训练
            generator_loss = self.generator_trainer.train_on_batch([real_images, real_images, real_index], None)



if __name__ == '__main__':
    iterations = int(1e5)
    ae_gan = AttentionGAN(train_data_path=r'traffic_data/filtered_data_all_005.csv')
    ae_gan.train(iterations)
