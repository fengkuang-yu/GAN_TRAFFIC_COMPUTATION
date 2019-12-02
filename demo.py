import os
import logging

from multiprocessing import Pool
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Add, Input, Lambda, Activation, Dense, Flatten, Reshape
from keras.layers import Bidirectional, CuDNNLSTM, TimeDistributed
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

train_data = np.random.random(size=(1000, 100))
train_label = np.random.random(size=(1000, 1))
optimizer = Adam()

input_layer = Input(shape=(100,))
hidden_layer = Dense(100, name='hidden_layer', activation='sigmoid')(input_layer)
output_layer = Dense(1,)(hidden_layer)

model = Model(input_layer, output_layer)
model.compile(optimizer, mean_squared_error)

model.fit(train_data, train_label, batch_size=64, epochs=10)
weight = model.get_layer(name='hidden_layer')

model.get_weights()
