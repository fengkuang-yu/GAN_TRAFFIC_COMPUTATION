#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/6/20 20:19
# software: PyCharm
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/6/12 18:51
# software: PyCharm
# ! -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VAE:
    def __init__(self):
        self.batch_size = 100
        self.original_dim = 784
        self.latent_dim = 10  # 隐变量取2维只是为了方便后面画图
        self.intermediate_dim = 256
        self.epochs = 100
        
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        real_x = Input(shape=(self.original_dim,))
        latent = self.encoder(real_x)
        fake_x = self.decoder(latent)
        
        self.vae = Model(real_x, fake_x)
        xent_loss = K.sum(K.binary_crossentropy(real_x, fake_x), axis=-1)
        
        self.vae.add_loss(xent_loss)
        self.vae.compile(optimizer='rmsprop')
        self.vae.summary()
    
    def build_encoder(self):
        x = Input(shape=(self.original_dim,))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        latent = Dense(self.latent_dim)(h)
        return Model(x, latent)
    
    def build_decoder(self):
        z = Input(shape=(self.latent_dim,))
        # 解码层，也就是生成器部分
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        # 建立模型
        return Model(z, x_decoded_mean)
    
    def train(self):
        # 加载MNIST数据集
        (x_train, y_train_), (x_test, y_test_) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        
        self.vae.fit(x_train,
                     shuffle=True,
                     epochs=self.epochs,
                     batch_size=self.batch_size,
                     validation_data=(x_test, None),
                     verbose=2)
     
    def hidden_distribution(self):
        # 构建encoder，然后观察各个数字在隐空间的分布
        # 仅仅对隐藏节点为二维节点有效
        (x_train, y_train_), (x_test, y_test_) = mnist.load_data()
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        x_test_encoded = self.encoder.predict(x_test, batch_size=self.batch_size)[0]
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
        plt.colorbar()
        plt.savefig(r'input_with_hidden.pdf')
        
        # 观察隐变量的两个维度变化是如何影响输出结果的
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        
        # 用正态分布的分位数来构建隐变量对
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit
        
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(r'hidden_with_output.pdf')
    
    def plot_test(self, dir, plot_real=False):
        (x_train, y_train_), (x_test, y_test_) = mnist.load_data()
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        gen_imgs = self.vae.predict(x_test, batch_size=self.batch_size)
        gen_imgs = gen_imgs.reshape(-1, 28, 28, 1)
        
        r, c = 10, 10
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(dir + "generated_image.pdf")
        plt.close()
        if plot_real == True:
            r, c = 10, 10
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(x_test[cnt, :].reshape(28, 28), cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(dir + "real_image.png")
            plt.close()


if __name__ == '__main__':
    dir = r'traffic_baselines/VAE/generated_imgs/'
    ccvae = VAE()
    ccvae.train()
    
    ccvae.plot_test(dir)
    # ccvae.hidden_distribution()

