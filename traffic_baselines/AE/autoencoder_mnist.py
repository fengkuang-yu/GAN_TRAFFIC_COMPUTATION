#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/7/14 16:14
# software: PyCharm

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Lambda, LeakyReLU
from keras.layers import Conv2D, ZeroPadding2D, Flatten
from keras.layers import UpSampling2D, Activation, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import mnist


class autoencoder():
    def __init__(self):
        self.batch_size = 100
        self.cols = 28
        self.rows = 28
        self.channels = 1
        self.img_shape = (28, 28, 1)
        self.latent_dim = 100
        self.input_shape = 28 * 28 * 1
        self.output_shape = 28 * 28 * 1
        self.epochs = 10000
        self.optimizer = Adam()
        
        self.ae = self.build_model()
    
    def build_model(self):
        input_layer = Input((self.input_shape,))
        latent_layer = Dense(self.latent_dim,)(input_layer)
        latent_layer_relu = LeakyReLU()(latent_layer)
        output_layer = Dense(self.output_shape,)(latent_layer_relu)
        
        ae = Model(input_layer, output_layer)
        xent_loss = K.mean(K.binary_crossentropy(output_layer, input_layer))
        ae.add_loss(xent_loss)
        ae.compile(optimizer=self.optimizer, )
        return ae
    
    def train(self):
        (x_train, y_train_), (_, _) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    
        for epoch in tqdm(range(self.epochs)):
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_imgs = x_train[idx]
            real_imgs = real_imgs.reshape(-1, self.input_shape)
            loss = self.ae.train_on_batch(real_imgs, None)
        
            if epoch % 200 == 0:
                print('loss:{}'.format(loss))


if __name__=='__main__':
    ae = autoencoder()
    ae.train()

