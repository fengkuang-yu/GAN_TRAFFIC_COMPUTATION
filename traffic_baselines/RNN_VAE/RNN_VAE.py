#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/6/20 17:19
# software: PyCharm
# !/usr/bin/env python
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

from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Bidirectional
from keras.layers import LSTM, BatchNormalization, TimeDistributed, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.datasets import mnist
from keras.utils import plot_model

model = Sequential()
input_shape = (149,40)
model.add(LSTM(units=20,return_sequences=True, input_shape=input_shape))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
# model.add(Dense(1, activation='sigmoid'))

# LSTM参数个数计算：ht-1与xt拼接、隐藏单元数、四个门的bias
#                    （20+40）*units*4+20*4
#
#
batch_size = 64
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_training, Y_training,
          batch_size=batch_size,
          epochs=30,
          validation_data=(x_test, y_test),
          verbose=1)
