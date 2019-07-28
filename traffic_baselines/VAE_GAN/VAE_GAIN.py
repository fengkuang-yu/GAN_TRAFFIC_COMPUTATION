#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/6/24 8:58
# software: PyCharm

from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, LeakyReLU, Dropout, Flatten, Reshape
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import RMSprop
from keras.datasets import mnist
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VAE_GAIN:
    def __init__(self):
        self.batch_size = 100
        self.original_dim = 784
        self.cols = 28
        self.rows = 28
        self.channels = 1
        self.img_shape = (self.rows, self.cols, self.channels)
        self.latent_dim = 100
        self.intermediate_dim = 256
        self.gamma1 = 1
        self.gamma2 = 1000
        self.lambda_d_penalty = 10
        self.epochs = 5000
        self.n_critic = 5
        self.optimizer = RMSprop()
        
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
        masks = Input(shape=self.img_shape, name='masks')
        z_p = Input(shape=(self.latent_dim,), name='random_distribution')
        
        masked_x = Lambda(lambda x: x[0] * x[1], output_shape=self.img_shape, )([real_x, masks])
        z_mean, z_log_var = self.encoder(masked_x)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='re-sampling_layer')([z_mean, z_log_var])
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        
        reconstruct_x = self.decoder(z)
        gen_x = self.decoder(z_p)
        interpolated_img = Lambda(self.random_weight_average, output_shape=self.img_shape,
                                  name='random_weight_average')([real_x, reconstruct_x])
        imputed_img = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2], output_shape=self.img_shape,
                             name='imputation_layer')([masks, real_x, reconstruct_x])
        real_valid, real_llike = self.critic(real_x)
        rec_valid, rec_llike = self.critic(imputed_img)
        gen_valid, gen_llike = self.critic(gen_x)
        inter_valid, inter_llike = self.critic(interpolated_img)
        
        reconstruct_loss = K.mean(K.square(reconstruct_x - real_x))
        reconstruct_llike_loss = self.mean_gaussian_negative_log_likelihood(real_llike, rec_llike)
        decoder_valid_loss = self.wasserstein_loss(-np.ones((self.batch_size, 1)), rec_valid)
        discriminate_loss = Lambda(self.critic_loss, output_shape=(1,),
                                   name='discriminate_loss')([interpolated_img, inter_valid,
                                                              gen_valid, rec_valid, real_valid, ])
        
        ### 构建VAE_GAN的编码部分 ###
        self.encoder.trainable = True
        self.decoder.trainable = False
        self.critic.trainable = False
        self.encoder_trainer = Model([real_x, masks], [imputed_img, rec_llike], name='encoder')
        self.encoder_trainer.add_loss(kl_loss + reconstruct_llike_loss)
        self.encoder_trainer.compile(optimizer=self.optimizer)
        # self.encoder.summary()
        
        ### 构建VAE_GAN的生成部分 ###
        self.encoder.trainable = False
        self.decoder.trainable = True
        self.critic.trainable = False
        self.decoder_trainer = Model([real_x, masks, z_p], [rec_llike, rec_valid, discriminate_loss], name='decoder')
        self.decoder_trainer.add_loss(self.gamma1 * reconstruct_llike_loss - self.gamma2 * discriminate_loss)
        self.decoder_trainer.compile(optimizer=self.optimizer)
        # self.decoder_trainer.summary()
        
        ### 构建GAN的判别部分
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.critic.trainable = True
        self.critic_trainer = Model([real_x, masks, z_p], [discriminate_loss], name='critic')
        self.critic_trainer.add_loss(discriminate_loss)
        self.critic_trainer.compile(optimizer=self.optimizer, )
        # self.critic_trainer.summary()
    
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
        
        # 算p(Z|X)的均值和方差
        z_mean = Dense(self.latent_dim)(fc1)
        z_log_var = Dense(self.latent_dim)(fc1)
        return Model(img, [z_mean, z_log_var])
    
    def build_decoder(self):
        model = Sequential()
        
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
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
        
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return Model(noise, img)
    
    def build_critic(self):
        
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
        valid = Dense(1)(fc1)
        
        return Model(img, [valid, conv3_output])
    
    def train(self):
        # 加载MNIST数据集
        (x_train, y_train_), (_, _) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        
        for epoch in tqdm(range(self.epochs)):
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_imgs = x_train[idx]
            real_imgs = real_imgs.reshape(-1, *self.img_shape)
            masks = self.mask_randomly(real_imgs)
            z_p = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            
            critic_loss = self.critic_trainer.train_on_batch([real_imgs, masks, z_p], None)
            encoder_loss = self.encoder_trainer.train_on_batch([real_imgs, masks], None)
            decoder_loss = self.decoder_trainer.train_on_batch([real_imgs, masks, z_p], None)
            
            if epoch % 200 == 0:
                print('encoder_loss:{};decoder_loss:{};critic_loss:{}'.format(encoder_loss, decoder_loss, critic_loss))
                self.plot_test('epoch:{}.png'.format(epoch))
    
    def test(self):
        pass
    
    def random_weight_average(self, args):
        x_real, x_rec = args
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * x_real) + ((1 - alpha) * x_rec)
    
    def critic_loss(self, args):
        interpolated_img, validity_interpolated, random_gen_x_valid, reconstruct_x_valid, real_x_valid, = args
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        
        rec_valid_loss = self.wasserstein_loss(fake, reconstruct_x_valid)
        real_valid_loss = self.wasserstein_loss(valid, real_x_valid)
        gen_valid_loss = self.wasserstein_loss(fake, random_gen_x_valid)
        penalty_loss = self.gradient_penalty_loss(None, validity_interpolated, interpolated_img)
        
        return rec_valid_loss + 2 * real_valid_loss + gen_valid_loss + self.lambda_d_penalty * penalty_loss
    
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
    
    def mean_gaussian_negative_log_likelihood(self, y_true, y_pred):
        # 实际上就是mse
        nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
        axis = tuple(range(1, len(K.int_shape(y_true))))
        return K.mean(K.sum(nll, axis=axis), axis=-1)
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    def patch_mask_randomly(self, images, mask_height=10, mask_width=10, ):
        """
        接收一个四维的矩阵，返回已经处理过的矩阵
        :param images:
        :param mask_height:
        :param mask_width:
        :return:
        """
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
    
    def mask_randomly(self, percentage=0.2):
        random_mask = np.random.random(size=(self.batch_size, self.rows, self.cols, self.channels))
        random_mask[random_mask > percentage] = 1
        random_mask[random_mask != 1] = 0
        return random_mask
    
    def plot_test(self, file_name, ):
        (_, _), (x_test, y_test_) = mnist.load_data()
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        x_test = x_test.reshape(-1, self.rows, self.cols, self.channels)[:100]
        masks = self.mask_randomly()
        corrupted = x_test * masks
        gen_imgs = self.encoder_trainer.predict([x_test, masks], batch_size=self.batch_size)[0]
        gen_imgs = gen_imgs.reshape(-1, self.rows, self.cols, self.channels)
        
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
        fig.savefig(os.path.join(os.getcwd(), 'generated_imgs', file_name))
        plt.close()


if __name__ == '__main__':
    dir = r'generated_image.png'
    vae_gan = VAE_GAIN()
    vae_gan.train()
    
    vae_gan.plot_test(dir)
    # ccvae.hidden_distribution()
