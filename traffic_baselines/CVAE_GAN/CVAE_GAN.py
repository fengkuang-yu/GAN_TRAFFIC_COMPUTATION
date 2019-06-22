#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/6/12 18:51
# software: PyCharm

from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, LeakyReLU
from keras.layers import Flatten, Reshape, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.layers.merge import _Merge
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        self.latent_dim = 100
        self.intermediate_dim = 256
        
        self.gamma = 0.1
        self.penalty = 10
        self.lambda1 = 3
        self.lambda2 = 1
        self.lambda3 = 1e-3
        self.lambda4 = 1e-3
        
        self.epochs = 5000
        self.n_critic = 5
        self.n_attrs = 10
        self.optimizer = RMSprop()
        
        # 构建编码器，生成器和判别器
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.critic = self.build_critic()
        self.classifier = self.build_classifier()
        
        self.encoder_trainer = None
        self.decoder_trainer = None
        self.critic_trainer = None
        self.classifier_trainer = None
        
        self.build_model()
    
    def build_model(self):
        # Algorithm
        x_real = Input(shape=self.img_shape, name='real_image')
        c_input = Input(shape=(self.n_attrs,), name='condition_input')
        
        z_mean, z_log_var = self.encoder([x_real, c_input])
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='re-sampling_layer')([z_mean, z_log_var])
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        z_p = Input(shape=(self.latent_dim,), name='random_distribution')
        x_rec = self.decoder([z, c_input])
        inter_img = RandomWeightedAverage()([x_real, x_rec])
        
        x_gen = self.decoder([z_p, c_input])
        
        inter_valid, inter_llike = self.critic(inter_img)
        rec_valid, rec_d_llike = self.critic(x_rec)
        real_valid, real_d_llike = self.critic(x_real)
        gen_valid, gen_d_llike = self.critic(x_gen)
        
        rec_class, rec_c_llike = self.classifier(x_rec)
        real_class, real_c_llike = self.classifier(x_real)
        gen_class, gen_c_llike = self.classifier(x_gen)
        
        val_loss = Lambda(self.valid_loss, output_shape=(1,),
                          name='d_loss')([rec_valid, real_valid, gen_valid, inter_img, inter_valid])
        gen_loss = Lambda(self.generative_loss, output_shape=self.img_shape,
                          name='generative_loss')([x_real, x_rec, real_d_llike, rec_d_llike, real_c_llike, rec_c_llike])
        d_llike_loss = Lambda(self.mean_gaussian_negative_log_likelihood, output_shape=(1,),
                              name='mean_gaussian_negative_log_likelihood_D')([real_d_llike, gen_d_llike])
        c_llike_loss = Lambda(self.mean_gaussian_negative_log_likelihood, output_shape=(1,),
                              name='mean_gaussian_negative_log_likelihood_C')([real_c_llike, gen_c_llike])
        c_class_loss = Lambda(self.classifier_loss, output_shape=(self.n_attrs,),
                              name='classifier_loss')([c_input, real_class])
        
        ### build classifier ###
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.critic.trainable = False
        self.classifier.trainable = True
        self.classifier_trainer = Model([x_real, c_input], [c_class_loss], name='classifier')
        self.classifier_trainer.add_loss(c_class_loss)
        self.classifier_trainer.compile(optimizer=self.optimizer)
        self.classifier_trainer.summary()
        
        ### build discriminator ###
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.critic.trainable = True
        self.classifier.trainable = False
        self.critic_trainer = Model([x_real, c_input, z_p], [val_loss], name='discriminator')
        self.critic_trainer.add_loss(val_loss)
        self.critic_trainer.compile(optimizer=self.optimizer)
        # self.critic_trainer.summary()
        
        ### build decoder ###
        self.encoder.trainable = False
        self.decoder.trainable = True
        self.critic.trainable = False
        self.classifier.trainable = False
        self.decoder_trainer = Model([x_real, c_input, z_p],
                                     [x_rec, gen_loss, d_llike_loss, c_llike_loss], name='decoder')
        self.decoder_trainer.add_loss(
            self.lambda2 * gen_loss + self.lambda3 * d_llike_loss + self.lambda4 * c_llike_loss)
        self.decoder_trainer.compile(optimizer=self.optimizer)
        self.decoder_trainer.summary()
        
        ### bulid encoder ###
        self.encoder.trainable = True
        self.decoder.trainable = False
        self.critic.trainable = False
        self.classifier.trainable = False
        self.encoder_trainer = Model([x_real, c_input, z_p],
                                     [x_rec, gen_loss], name='encoder')
        self.encoder_trainer.add_loss(self.lambda1 * kl_loss + self.lambda2 * gen_loss)
        self.encoder_trainer.compile(optimizer=self.optimizer)
        self.encoder.summary()
    
    def valid_loss(self, args):
        rec_valid, real_valid, gen_valid, inter_imgs, inter_valid = args
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        
        rec_valid_loss = self.wasserstein_loss(rec_valid, fake)
        real_valid_loss = self.wasserstein_loss(real_valid, valid)
        gen_valid_loss = self.wasserstein_loss(gen_valid, fake)
        penalty_loss = self.gradient_penalty_loss(None, inter_valid, inter_imgs)
        
        return rec_valid_loss + 2 * real_valid_loss + gen_valid_loss + self.penalty * penalty_loss
    
    def generative_loss(self, args):
        x_real, x_rec, real_d_llike, rec_d_llike, real_c_llike, rec_c_llike = args
        x_loss = K.mean(K.square(x_real - x_rec))
        d_loss = K.mean(K.square(real_d_llike - rec_d_llike))
        c_loss = K.mean(K.square(real_c_llike - rec_c_llike))
        return x_loss + d_loss + c_loss
    
    def classifier_loss(self, args):
        c_input, real_class = args
        return K.mean(K.categorical_crossentropy(c_input, real_class))
    
    def build_encoder(self):
        img = Input(shape=self.img_shape)
        conv1 = Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(img)
        # conv1 = BatchNormalization(momentum=0.8)(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        
        conv2 = Conv2D(32, kernel_size=3, strides=2, padding="same")(conv1)
        conv2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(conv2)  # 判断是否需要填充
        # conv2 = BatchNormalization(momentum=0.8)(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        
        conv3 = Conv2D(64, kernel_size=3, strides=2, padding="same")(conv2)
        # conv3 = BatchNormalization(momentum=0.8)(conv3)
        conv3_output = LeakyReLU(alpha=0.2)(conv3)
        
        fc1 = Flatten()(conv3_output)
        c_input = Input(shape=(self.n_attrs,))
        fc_output = Concatenate(axis=-1)([fc1, c_input])
        
        # 算p(Z|X)的均值和方差
        z_mean = Dense(self.latent_dim)(fc_output)
        z_log_var = Dense(self.latent_dim)(fc_output)
        return Model([img, c_input], [z_mean, z_log_var])
    
    def build_decoder(self):
        model = Sequential()
        
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim+self.n_attrs))
        model.add(Reshape((7, 7, 128)))
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
        
        z_noise = Input(shape=(self.latent_dim,))
        c_input = Input(shape=(self.n_attrs,))
        noise = Concatenate()([z_noise, c_input])
        
        img = model(noise)
        
        return Model([z_noise, c_input], img)
    
    def build_critic(self):
        
        img = Input(shape=self.img_shape)
        conv1 = Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(img)
        conv1 = BatchNormalization(momentum=0.8)(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        
        conv2 = Conv2D(32, kernel_size=3, strides=2, padding="same")(conv1)
        conv2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(conv2)  # 判断是否需要填充
        # conv2 = BatchNormalization(momentum=0.8)(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        
        conv3 = Conv2D(64, kernel_size=3, strides=2, padding="same")(conv2)
        # conv3 = BatchNormalization(momentum=0.8)(conv3)
        conv3_output = LeakyReLU(alpha=0.2)(conv3)
        
        fc1 = Flatten()(conv3_output)
        valid = Dense(1)(fc1)
        
        return Model(img, [valid, conv3_output])
    
    def build_classifier(self):
        img = Input(shape=self.img_shape)
        
        conv1 = Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(img)
        conv1 = BatchNormalization(momentum=0.8)(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        
        conv2 = Conv2D(32, kernel_size=3, strides=2, padding="same")(conv1)
        conv2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(conv2)  # 判断是否需要填充
        conv2 = BatchNormalization(momentum=0.8)(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        
        conv3 = Conv2D(64, kernel_size=3, strides=2, padding="same")(conv2)
        conv3 = BatchNormalization(momentum=0.8)(conv3)
        conv3_output = LeakyReLU(alpha=0.2)(conv3)
        
        fc1 = Flatten()(conv3_output)
        
        x = Dense(1024)(fc1)
        x = Activation('relu')(x)
        
        x = Dense(self.n_attrs)(x)
        x_out = Activation('softmax')(x)
        
        return Model(img, [x_out, fc1])
    
    def train(self):
        # 加载MNIST数据集
        (x_train, y_train_), (_, _) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        
        # 随机生隐空间的噪声数据，模拟生成数据的过程
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))  # Dummy gt for gradient penalty
        
        for epoch in tqdm(range(self.epochs)):
            # Train autoencoder
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_imgs = x_train[idx]
            real_imgs = real_imgs.reshape(-1, *self.img_shape)
            z_p = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            critic_loss, _, _, _, _ = self.critic_trainer.train_on_batch([real_imgs, z_p], [dummy, fake, fake, valid])
            
            encoder_loss = self.encoder_trainer.train_on_batch(real_imgs, None)
            decoder_loss = self.decoder_trainer.train_on_batch(real_imgs, None)
            
            if epoch % 500 == 0:
                # print('encoder_loss:{};decoder_loss:{}'.format(encoder_loss, decoder_loss))
                print('encoder_loss:{};decoder_loss:{};critic_loss:{}'.format(encoder_loss, decoder_loss, critic_loss))
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
    
    def mean_gaussian_negative_log_likelihood(self, args):
        y_true, y_pred = args
        # 实际上就是mse
        nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
        axis = tuple(range(1, len(K.int_shape(y_true))))
        return K.mean(K.sum(nll, axis=axis), axis=-1)
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    def plot_test(self, file_name, plot_real=False):
        (_, _), (x_test, y_test_) = mnist.load_data()
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        x_test = x_test.reshape(-1, self.rows, self.cols, self.channels)
        gen_imgs = self.encoder_trainer.predict(x_test, batch_size=self.batch_size)[0]
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
            fig.savefig(dir + "real_image.png")
            plt.close()


if __name__ == '__main__':
    dir = r'generated_image.png'
    vae_gan = VAE_GAN()
    vae_gan.train()
    
    vae_gan.plot_test(dir)
    # ccvae.hidden_distribution()
