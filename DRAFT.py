#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/6/12 18:51
# software: PyCharm
# ! -*- coding: utf-8 -*-

'''用Keras实现的VAE
   目前只保证支持Tensorflow后端
   改写自
   https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

import os
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Lambda, LeakyReLU, Dropout, Flatten, Reshape
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.utils import plot_model
from functools import partial
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class KLLossLayer(Layer):
    __name__ = 'kl_loss_layer'
    
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(KLLossLayer, self).__init__(**kwargs)
    
    def lossfun(self, z_avg, z_log_var):
        kl_loss = -0.5 * K.mean(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var))
        return 3 * kl_loss
    
    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        loss = self.lossfun(z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)
        
        return z_avg


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    
    def _merge_function(self, inputs):
        alpha = K.random_uniform((100, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class VAE_GAN:
    def __init__(self):
        self.batch_size = 100
        self.original_dim = 784
        self.cols = 28
        self.rows = 28
        self.channels = 1
        self.img_shape = (self.rows, self.cols, self.channels)
        self.latent_dim = 10  # 隐变量取2维只是为了方便后面画图
        self.intermediate_dim = 256
        self.epochs = 10000
        self.n_critic = 5
        self.optimizer = Adam()
        
        # 构建编码器，生成器和判别器
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.critic = self.build_critic()
        
        self.encoder_trainer = None
        self.decoder_trainer = None
        self.critic_trainer = None
        
        self.build_model()
    
    def build_model(self):
        real_x = Input(shape=self.img_shape, name='real_image')
        z_mean, z_log_var = self.encoder(real_x)
        
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='re-sampling_layer')([z_mean, z_log_var])
        # kl_loss = Lambda()
        
        reconstruct_x_flatten = self.decoder(z)
        reconstruct_x = Reshape(self.img_shape)(reconstruct_x_flatten)
        interpolated_img = RandomWeightedAverage()([real_x, reconstruct_x])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)
        partial_gp_loss_rec = partial(self.gradient_penalty_loss, averaged_samples=interpolated_img)
        partial_gp_loss_rec.__name__ = 'gradient_penalty'  # Keras requires function names
        
        z_p = Input(shape=(self.latent_dim,), name='random_distribution')
        gen_x_flatten = self.decoder(z_p)
        gen_x = Reshape(self.img_shape)(gen_x_flatten)
        random_gen_x_valid = self.critic(gen_x)
        reconstruct_x_valid = self.critic(reconstruct_x)
        real_x_valid = self.critic(real_x)
        
        ### 构建VAE_GAN的编码部分 ###
        self.encoder.trainable = True
        self.decoder.trainable = False
        self.encoder_trainer = Model(inputs=real_x, outputs=reconstruct_x, name='encoder')
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        reconstruct_loss = K.sum(K.binary_crossentropy(real_x, reconstruct_x),
                                 axis=np.arange(1, len(reconstruct_x.shape)))
        
        encoder_loss = K.mean(reconstruct_loss + kl_loss)
        self.encoder_trainer.add_loss(encoder_loss)
        self.encoder_trainer.compile(optimizer=self.optimizer)
        # self.encoder.summary()
        
        ### 构建VAE_GAN的生成部分 ###
        self.encoder.trainable = False
        self.decoder.trainable = True
        self.critic.trainable = False
        
        self.decoder_trainer = Model(inputs=[real_x],
                                     outputs=[reconstruct_x],
                                     name='decoder')
        self.decoder_trainer.compile(loss=['binary_crossentropy',],
                                     optimizer=self.optimizer,)
        # self.decoder_trainer.summary()
        
        ### 构建GAN的判别部分
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.critic.trainable = True
        self.critic_trainer = Model(inputs=[real_x, z_p],
                                    outputs=[validity_interpolated,
                                             random_gen_x_valid,
                                             reconstruct_x_valid,
                                             real_x_valid, ],
                                    name='critic')
        self.critic_trainer.compile(loss=[partial_gp_loss_rec,
                                          self.wasserstein_loss,
                                          self.wasserstein_loss,
                                          self.wasserstein_loss, ],
                                    optimizer=self.optimizer,
                                    loss_weights=[10, 1, 1, 1])
        # self.critic_trainer.summary()
    
    def build_encoder(self):
        x = Input(shape=self.img_shape, )
        x_flatten = Flatten()(x)
        h = Dense(self.intermediate_dim, activation='relu')(x_flatten)
        
        # 算p(Z|X)的均值和方差
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)
        return Model(x, [z_mean, z_log_var])
    
    def build_decoder(self):
        z = Input(shape=(self.latent_dim,))
        # 解码层，也就是生成器部分
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        # 建立模型
        return Model(z, x_decoded_mean)
    
    def build_critic(self):
        
        model = Sequential()
        
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        
        img = Input(shape=self.img_shape)
        validity = model(img)
        
        return Model(img, validity)
    
    def train(self):
        # 加载MNIST数据集
        (x_train, y_train_), (x_test, y_test_) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        
        # 随机生隐空间的噪声数据，模拟生成数据的过程
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))  # Dummy gt for gradient penalty
        
        for epoch in tqdm(range(self.epochs)):
            # Train autoencoder
            
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_imgs = x_train[idx]
            real_imgs = real_imgs.reshape(-1, *self.img_shape)
            # z_p = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            # critic_loss, _, _, _, _ = self.critic_trainer.train_on_batch([real_imgs, z_p], [dummy, fake, fake, valid])
            
            encoder_loss = self.encoder_trainer.train_on_batch(real_imgs, None)
            decoder_loss = self.decoder_trainer.train_on_batch([real_imgs], [real_imgs])
            
            if epoch % 5000 == 0:
                print('encoder_loss:{};decoder_loss:{}'.format(encoder_loss, decoder_loss))
                # print('encoder_loss:{};decoder_loss:{};critic_loss:{}'.format(encoder_loss, decoder_loss, critic_loss))
                self.plot_test('epoch:{}.png'.format(epoch))
    
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
    
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def zero_loss(self, y_true, y_pred):
        """
        args:
            y_true():
            y_pred():
        """
        return K.zeros_like(y_true)
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    def plot_test(self, file_name, plot_real=False):
        (x_train, y_train_), (x_test, y_test_) = mnist.load_data()
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        x_test = x_test.reshape(-1, self.rows, self.cols, self.channels)
        gen_imgs = self.encoder_trainer.predict(x_test, batch_size=self.batch_size)
        gen_imgs = gen_imgs.reshape(-1, 28, 28, 1)
        
        r, c = 10, 10
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(os.getcwd(), 'generated_imgs', file_name))
        plt.close()
        if plot_real == True:
            r, c = 10, 10
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(x_test[cnt, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig('generated_imgs/' + "real_image.png")
            plt.close()


if __name__ == '__main__':
    dir = r'generated_image.png'
    ccvae = VAE_GAN()
    ccvae.train()
    
    ccvae.plot_test(dir, plot_real=True)
    # ccvae.hidden_distribution()
