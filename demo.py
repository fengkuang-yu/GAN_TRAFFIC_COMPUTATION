# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   demo.py
@Time    :   2019/9/6 19:28
@Desc    :
"""
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/10/16 15:30
# software: PyCharm

import math
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Add, Input, Lambda, Activation
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
K.set_session(session)

if 'TF_KERAS' in os.environ and os.environ['TF_KERAS'] != '0':
    from tensorflow.python import keras

    TF_KERAS = True
else:
    import keras

    TF_KERAS = False


class LayerNormalization(keras.layers.Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class PositionEmbedding(keras.engine.Layer):

    def __init__(self, size=None, mode='sum', min_timescale=1.0, max_timescale=1.0e4, start_index=0, **kwargs):
        self.size = size
        self.mode = mode
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.start_index = start_index

        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        min_timescale, max_timescale, start_index = self.min_timescale, self.max_timescale, self.start_index
        batch_size, length, channels = x.shape
        position = K.cast(K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1 + start_index, float)
        position_i = K.expand_dims(position, 2)
        num_timescales = channels // 2
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) / (K.cast(num_timescales, float) - 1))
        inv_timescales = min_timescale * K.exp(K.cast(K.arange(num_timescales), float) * -log_timescale_increment)
        position_j = K.expand_dims(inv_timescales, 0)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.sin(position_ij), K.cos(position_ij)], 2)
        position_ij = tf.pad(position_ij, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])

        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer.

    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.

    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.

    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.

    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 dropout_rate=0.0,
                 **kwargs):
        self.supports_masking = True
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))

            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y


class ScaledDotProductAttention(keras.layers.Layer):
    r"""The attention layer that takes three inputs representing queries, keys and values.

    \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(0, key_len), axis=0)
            upper = K.expand_dims(K.arange(0, query_len), axis=-1)
            e *= K.expand_dims(K.cast(indices <= upper, K.floatx()), axis=0)
        if mask is not None:
            e *= K.cast(K.expand_dims(mask, axis=-2), K.floatx())
        a = e / (K.sum(e, axis=-1, keepdims=True) + K.epsilon())
        v = K.batch_dot(a, value)
        if self.return_attention:
            return [v, a]
        return v


class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        y = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        y = self._reshape_from_batches(y, self.head_num)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        if TF_KERAS:
            # Add shape information to tensor when using `tf.keras`
            input_shape = [K.int_shape(q), K.int_shape(k), K.int_shape(v)]
            output_shape = self.compute_output_shape(input_shape)
            if output_shape[1] is not None:
                output_shape = (-1,) + output_shape[1:]
                y = K.reshape(y, output_shape)
        return y


class attention_mnist_AE():
    def __init__(self):
        # 设置训练超参数
        self.batch_size = 100
        self.cols = 60
        self.rows = 60
        self.img_shape = (self.rows, self.cols)
        self.multi_head_num = 6
        self.ff_hidden_unit_num = 1024
        self.epochs = 10000
        self.optimizer = Adam(beta_1=0.9, beta_2=0.98, epsilon=10e-9)

        # 构建模型
        self.attention_model = self.build_attention_model_with_single_input()

        # 数据处理相关
        self.missing_percentage = 0.2
        self.test_percent = 0.1
        self.scalar = MinMaxScaler()
        self.data = self.load_data(r'traffic_data/data_all.csv')
        self.index = np.array([[x] for x in range(len(self.data))])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.index, test_size=self.test_percent, random_state=12)
        self.miss_mode = 'spatial_line'

    def build_attention_model_with_single_input(self):

        input_x = Input(shape=self.img_shape, name='real_inputs')
        masks = Input(shape=self.img_shape, name='mask_inputs')
        masked_x = Lambda(lambda x: x[0] * x[1], output_shape=self.img_shape, name='masked_input')([input_x, masks])

        pos_encoding_1 = PositionEmbedding()(masked_x)
        add_position_1 = Add()([masked_x, pos_encoding_1])

        attention_multi_1 = MultiHeadAttention(head_num=self.multi_head_num)(add_position_1)
        add_attention_1 = Add()([add_position_1, attention_multi_1])
        layer_normal_1 = LayerNormalization()(add_attention_1)
        feed_forward_1 = FeedForward(units=self.ff_hidden_unit_num)(layer_normal_1)
        feed_forward_res_1 = Add()([feed_forward_1, layer_normal_1])

        attention_multi_2 = MultiHeadAttention(head_num=self.multi_head_num)(feed_forward_res_1)
        add_attention_2 = Add()([feed_forward_res_1, attention_multi_2])
        layer_normal_2 = LayerNormalization()(add_attention_2)
        feed_forward_2 = FeedForward(units=self.ff_hidden_unit_num)(layer_normal_2)
        feed_forward_res_2 = Add()([feed_forward_2, layer_normal_2])

        attention_multi_3 = MultiHeadAttention(head_num=self.multi_head_num)(feed_forward_res_2)
        add_attention_3 = Add()([feed_forward_res_2, attention_multi_3])
        layer_normal_3 = LayerNormalization()(add_attention_3)
        feed_forward_3 = FeedForward(units=self.ff_hidden_unit_num)(layer_normal_3)
        feed_forward_3 = Add()([feed_forward_3, layer_normal_3])
        fake_res = Activation('sigmoid')(feed_forward_3)

        imputed_img = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2], output_shape=self.img_shape,
                             name='imputation_layer')([masked_x, input_x, fake_res])

        attention_model = Model([input_x, masks], [fake_res, imputed_img], name='basic_generator')
        att_loss = K.mean(mean_squared_error(input_x, fake_res))
        attention_model.add_loss(att_loss)
        attention_model.compile(optimizer=self.optimizer, )
        attention_model.summary()

        return attention_model

    def train(self):
        # x_train = self.x_train
        x_train = self.X_train

        for epoch in tqdm(range(self.epochs)):
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_images = x_train[idx]
            real_images = real_images.reshape((-1, *self.img_shape))
            masks = self.mask_randomly(real_images.shape, mode=self.miss_mode)
            loss = self.attention_model.train_on_batch([real_images, masks], None)

            if epoch % 1000 == 0:
                self.plot_test_mask(
                    self.miss_mode + '{:.1%}_{:0>5d}epochs_gen.png'.format(self.missing_percentage, epoch)
                )
                print('loss:{}'.format(loss))

    def plot_test(self, file_name=None):

        img_index = np.random.randint(0, self.X_test.shape[0], 100)
        x_test = self.X_test[img_index].reshape(-1, *self.img_shape)
        x_gen = self.attention_model.predict(x_test)

        r, c = 4, 4
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(x_gen[cnt, :, :], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.subplots_adjust(wspace=0.9, hspace=0.9)
        fig.suptitle('Generated images')
        # fig.savefig(os.path.join(os.getcwd(), 'generated_imgs', file_name))
        plt.show()
        plt.close()

        r, c = 4, 4
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(x_test[cnt, :, :], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.subplots_adjust(wspace=0.9, hspace=0.9)
        fig.suptitle('Real images')
        # fig.savefig(os.path.join(os.getcwd(), 'generated_imgs', file_name))
        plt.show()
        plt.close()

    def mask_randomly(self, shape, mode='patch'):
        """
        接收一个三维以上的矩阵，返回已经处理过的矩阵
        :param images:
        :param mode:
        :param percent:
        :param mask_height:
        :param mask_width:
        :return:
        """
        assert len(shape) >= 3
        assert isinstance(shape, tuple)
        percent = self.missing_percentage if self.missing_percentage else 0.1
        masks = np.ones(shape=shape)
        if mode == 'patch':
            img_rows, img_width = shape[1], shape[2]
            mask_height = mask_width = int(np.sqrt(img_rows * img_width * percent))

            assert img_rows - mask_height > 0
            assert img_width - mask_width > 0
            mask_rows_start = np.random.randint(0, img_rows - mask_height, shape[0])
            mask_rows_end = mask_rows_start + mask_height
            mask_cols_start = np.random.randint(0, img_width - mask_width, shape[0])
            mask_cols_end = mask_cols_start + mask_width

            for i in range(shape[0]):
                _y1, _y2, _x1, _x2 = mask_rows_start[i], mask_rows_end[i], mask_cols_start[i], mask_cols_end[i],
                masks[i][_y1:_y2, _x1:_x2] = 0
            return masks

        elif mode == 'spatial_line':
            for i in range(shape[0]):
                index_cols = np.random.randint(0, shape[2], int(percent * shape[2]))
                masks[i][:, index_cols] = 0
            return masks

        elif mode == 'temporal_line':
            for i in range(shape[0]):
                index_rows = np.random.randint(0, shape[1], int(percent * shape[1]))
                masks[i][index_rows, :] = 0
            return masks

        elif mode == 'random':
            pass

    def load_data(self, dir):
        # 从excel表中选出需要的数据
        traffic_data = pd.read_csv(dir, index_col=0)
        data = traffic_data.values[:, 80: 140]  # 选择数据
        time_steps = self.cols if self.cols else 1

        # 数据处理
        size = data.shape
        data = np.array(data)
        reshaped_data = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
        for i in range(size[0] - time_steps + 1):
            reshaped_data[i, :] = data[i:i + time_steps, :].flatten()

        # data = self.data_pro(data, time_steps=60)
        data = self.scalar.fit_transform(reshaped_data)
        data = data.reshape(-1, *self.img_shape)
        return data

    def plot_test_mask(self, file_name, ):
        x_test = self.X_test
        idx = np.random.randint(0, 100, self.batch_size)
        x_test = x_test[idx]
        masks = self.mask_randomly(x_test.shape, mode=self.miss_mode)
        corrupted = x_test * masks
        gen_imgs, masks_gen = self.attention_model.predict([x_test, masks], batch_size=self.batch_size)
        x_flatten = x_test.reshape((-1, self.rows * self.cols))
        g_flatten = gen_imgs.reshape((-1, self.rows * self.cols))
        m_flatten = masks_gen.reshape((-1, self.rows * self.cols))
        x_flatten = self.scalar.inverse_transform(x_flatten)
        g_flatten = self.scalar.inverse_transform(g_flatten)
        m_flatten = self.scalar.inverse_transform(m_flatten)

        # 生成样本的MAPE值
        # total_mape = np.sum(np.abs(g_flatten - x_flatten) / x_flatten) / np.prod(x_flatten.shape)
        total_rmse = np.sqrt(np.sum(np.power(g_flatten - x_flatten, 2)) / np.prod(x_flatten.shape))
        total_mae = np.sum(np.abs(g_flatten - x_flatten)) / np.prod(x_flatten.shape)

        # 填充部分的MAPE值
        # masks_mape = np.sum(np.abs(m_flatten - x_flatten) / x_flatten) / np.sum(1 - masks)
        masks_rmse = np.sqrt(np.sum(np.power(g_flatten - x_flatten, 2)) / np.sum(1 - masks))
        masks_mae = np.sum(np.abs(m_flatten - x_flatten)) / np.sum(1 - masks)

        r, c = 2, 8
        fig, axs = plt.subplots(r * 3 + 1, c)
        for j in range(c):
            for index, temp in enumerate([x_test, corrupted, gen_imgs]):
                axs[index, j].imshow(temp[j, :, :], cmap='gray')
                axs[index, j].axis('off')
        for j in range(c):
            axs[3, j].axis('off')
        for j in range(c):
            for index, temp in enumerate([x_test, corrupted, gen_imgs]):
                axs[4 + index, j].imshow(temp[c + j, :, :], cmap='gray')
                axs[4 + index, j].axis('off')
        fig.suptitle('total_rmse:{:.3f};total_mae:{:.3f}\n'
                     'masks_rmse:{:.3f};masks_mae:{:.3f}'.format(total_rmse, total_mae, masks_rmse, masks_mae))
        fig.savefig(os.path.join(os.getcwd(), 'generated_images', file_name), dpi=300)
        plt.close()


if __name__ == '__main__':
    ae = attention_mnist_AE()
    ae.train()
    ae.plot_test_mask('gen.png')
    ae.attention_model.save('att_ae_{}_{}_{:.1%}.h5'.format(ae.miss_mode, ae.epochs, ae.missing_percentage))
