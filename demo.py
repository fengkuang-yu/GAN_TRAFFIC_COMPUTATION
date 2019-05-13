# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   demo.py
@Time    :   2019/4/19 16:26
@Desc    :
"""

import os
from collections import namedtuple  # 使用namedtuple存放神经网络的超参数

import keras
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.examples.tutorials.mnist import input_data


def show_batch_images(images):
    """
    Plot show a batch figures
    :param images: a batch size figures
    :return:
    """
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtN = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtImg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtN, sqrtN))
    gs = gridspec.GridSpec(sqrtN, sqrtN)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtImg, sqrtImg]), cmap='Greys_r')
    plt.show()
    return fig


def corrupt_images(images, random_miss_percent=0., average_miss=1):
    """
    corrupt images
    :param images:
    :param random_miss_percent:
    :param average_missing:
    :return: corrupted images
    """
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrt_n = int(np.ceil(np.sqrt(images.shape[0])))
    sqrt_img = int(np.ceil(np.sqrt(images.shape[1])))

    corrupted_images = np.copy(images)  # make a copy of images

    fig = plt.figure(figsize=(sqrt_n, sqrt_n))
    gs = gridspec.GridSpec(sqrt_n, sqrt_n)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        pass
    return corrupted_images


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type, param):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='validation')
        plt.grid(True)
        plt.xlabel(loss_type, fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.legend(loc="upper right", fontsize=14)
        plt.savefig(os.path.join(param.file_path, 'figure\\',
                                 'loss_ST={}_{}_pred_time{}.pdf'.format(param.loop_num,
                                                                        param.time_intervals,
                                                                        5 * (param.predict_intervals + 1))))


class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', # cross-entropy loss
                                   optimizer=optimizer, # adam optimizer
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        # print network structure
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
        # (X_train, _), (_, _) = mnist.load_data()
        X_train = mnist.train.images

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = X_train.reshape(-1, 28, 28, 1)
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("generated_images/GAN_epoch%d.png" % epoch)
        plt.close()

    def save_model(self):
        self.discriminator.save(r'saved_model\GAN_discriminator.h5')
        self.generator.save(r'saved_model\GAN_generator.h5')

class WGAN(GAN):
    def train(self, epochs, batch_size=128, sample_interval=50):
        pass

    pass


class WGAN_GP(GAN):
    def train(self, epochs, batch_size=128, sample_interval=50):
        pass

    pass

NetParam = namedtuple('NetParam', ['MODEL',  # dcgan, wgan, or wgan-gp
                                   'DIM',  # dimensions
                                   'BATCH_SIZE',  # Batch size
                                   'CRITIC_ITERS',  # For WGAN and WGAN-GP, number of critic iters per gen iter
                                   'LAMBDA',  # Gradient penalty lambda hyperparameter
                                   'ITERS',  # How many generator iterations to train for
                                   'OUTPUT_DIM'  # Number of pixels in MNIST (28*28)
                                   ])

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=100, batch_size=128, sample_interval=200)
    gan.save_model()
    #
    # # read mnist and set hyper-parameters
    # mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    # x_train, y_train = mnist.train.next_batch(64)
    #
    # # define a hyper-parameter namedtuple
    # gan = GAN()
    # gan.train(epochs=30000, batch_size=32, sample_interval=200)
