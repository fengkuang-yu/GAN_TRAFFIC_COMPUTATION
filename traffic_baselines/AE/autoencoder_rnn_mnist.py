#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/9/4 17:16
# software: PyCharm

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.layers import Input, RepeatVector, CuDNNGRU, Bidirectional, GRU
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import mnist


class autoencoder_rnn():
    def __init__(self):
        self.batch_size = 128
        self.cols = 28
        self.rows = 28
        self.channels = 1
        self.img_shape = (28, 28, 1)
        self.latent_dim = 100
        self.input_shape = (28, 28)
        self.output_shape = (28, 28)
        self.epochs = 20
        self.optimizer = Adam()
        self.ae = self.build_model()
        (self.x_train, self.y_train_), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
    
    def build_model(self):
        input_layer = Input(self.input_shape,)
        lstm_layer_1 = Bidirectional(CuDNNGRU(50, return_sequences=True,))(input_layer)
        lstm_layer_2 = Bidirectional(CuDNNGRU(50,))(lstm_layer_1)
        latent_layer = RepeatVector(self.cols)(lstm_layer_2)
        output_layer = CuDNNGRU(self.cols, return_sequences=True, )(latent_layer)
        
        ae = Model(input_layer, output_layer)
        xent_loss = K.mean(K.binary_crossentropy(output_layer, input_layer))
        ae.add_loss(xent_loss)
        ae.compile(optimizer=self.optimizer, )
        return ae
    
    def train(self):
        x_train, y_train_ = self.x_train, self.y_train_
        # x_train = x_train.astype('float32') / 255.
        # x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
        
        # for epoch in tqdm(range(self.epochs)):
        #     idx = np.random.randint(0, x_train.shape[0], self.batch_size)
        #     real_imgs = x_train[idx]
        #     # real_imgs = real_imgs.reshape((-1, self.input_shape))
        #     loss = self.ae.train_on_batch(real_imgs, None)
        #
        #     if epoch % 1000 == 0:
        #         print('loss:{}'.format(loss))
        self.ae.fit(x_train, None, epochs=self.epochs, verbose=2)
    def plot_test(self, test_data, gen_data, file_name):
        plot_data_instance = (x for x in zip(test_data, gen_data))
        rows, columns = 2, 6
        fig, axs = plt.subplots(rows * 2, columns, figsize=(6, 6))
        for row in range(rows):
            for col in range(columns):
                temp = next(plot_data_instance)
                axs[row * 2, col].imshow(temp[0], cmap='gray')
                axs[row * 2, col].axis('off')
                axs[row * 2 + 1, col].imshow(temp[1], cmap='gray')
                axs[row * 2 + 1, col].axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.savefig(os.path.join(os.getcwd(), 'generated_imgs', file_name))
        plt.close()


if __name__ == '__main__':
    ae_rnn = autoencoder_rnn()
    ae_rnn.train()
    gen_data = ae_rnn.ae.predict(ae_rnn.x_test)
    
    ae_rnn.plot_test(ae_rnn.x_test, gen_data, 'demo.png')

