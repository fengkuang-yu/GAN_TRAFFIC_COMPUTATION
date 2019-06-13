import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import load_model

from .base import BaseModel
from .layers import *
from .utils import set_trainable, zero_loss, sample_normal, time_format


class ClassifierLossLayer(Layer):
    __name__ = 'classifier_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(ClassifierLossLayer, self).__init__(**kwargs)

    def lossfun(self, c_true, c_pred):
        return K.mean(keras.metrics.categorical_crossentropy(c_true, c_pred))

    def call(self, inputs):
        c_true = inputs[0]
        c_pred = inputs[1]
        loss = self.lossfun(c_true, c_pred)
        self.add_loss(loss, inputs=inputs)

        return c_true

class DiscriminatorLossLayer(Layer):
    __name__ = 'discriminator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake_f, y_fake_p):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_real)
        loss_real = keras.metrics.binary_crossentropy(y_pos, y_real)
        loss_fake_f = keras.metrics.binary_crossentropy(y_neg, y_fake_f)
        loss_fake_p = keras.metrics.binary_crossentropy(y_neg, y_fake_p)
        return K.mean(loss_real + loss_fake_f + loss_fake_p)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake_f = inputs[1]
        y_fake_p = inputs[2]
        loss = self.lossfun(y_real, y_fake_f, y_fake_p)
        self.add_loss(loss, inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    __name__ = 'generator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_r, x_f, f_D_x_f, f_D_x_r, f_C_x_r, f_C_x_f):
        loss_x = K.mean(K.square(x_r - x_f))
        loss_d = K.mean(K.square(f_D_x_r - f_D_x_f))
        loss_c = K.mean(K.square(f_C_x_r - f_C_x_f))

        return loss_x + loss_d + loss_c

    def call(self, inputs):
        x_r = inputs[0]
        x_f = inputs[1]
        f_D_x_r = inputs[2]
        f_D_x_f = inputs[3]
        f_C_x_r = inputs[4]
        f_C_x_f = inputs[5]
        loss = self.lossfun(x_r, x_f, f_D_x_r, f_D_x_f, f_C_x_r, f_C_x_f)
        self.add_loss(loss, inputs=inputs)

        return x_r

class FeatureMatchingLayer(Layer):
    __name__ = 'feature_matching_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(FeatureMatchingLayer, self).__init__(**kwargs)

    def lossfun(self, f1, f2):
        f1_avg = K.mean(f1, axis=0)
        f2_avg = K.mean(f2, axis=0)
        return 10**-3 * 0.5 * K.mean(K.square(f1_avg - f2_avg))

    def call(self, inputs):
        f1 = inputs[0]
        f2 = inputs[1]
        loss = self.lossfun(f1, f2)
        self.add_loss(loss, inputs=inputs)

        return f1

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

def discriminator_accuracy(x_r, x_f, x_p):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_r)
        x_neg = K.zeros_like(x_r)
        loss_r = K.mean(keras.metrics.binary_accuracy(x_pos, x_r))
        loss_f = K.mean(keras.metrics.binary_accuracy(x_neg, x_f))
        loss_p = K.mean(keras.metrics.binary_accuracy(x_neg, x_p))
        return (1.0 / 3.0) * (loss_r + loss_p + loss_f)

    return accfun

def generator_accuracy(x_p, x_f):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_p)
        loss_p = K.mean(keras.metrics.binary_accuracy(x_pos, x_p))
        loss_f = K.mean(keras.metrics.binary_accuracy(x_pos, x_f))
        return 0.5 * (loss_p + loss_f)

    return accfun

class CVAEGAN(BaseModel):
    def __init__(self,
        input_shape=(128, 128, 3),
        num_attrs=2,
        z_dims = 64,
        **kwargs
    ):
        super(CVAEGAN, self).__init__(input_shape=input_shape, **kwargs)

        self.input_shape = input_shape
        self.num_attrs = num_attrs
        self.z_dims = z_dims

        self.f_enc = None
        self.f_dec = None
        self.f_dis = None
        self.f_cls = None
        self.enc_trainer = None
        self.dec_trainer = None
        self.dis_trainer = None
        self.cls_trainer = None

        self.build_model()

    def train_on_batch(self, x_batch):
        x_r, c = x_batch

        batchsize = len(x_r)
        z_p = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')

        x_dummy = np.zeros(x_r.shape, dtype='float32')
        c_dummy = np.zeros(c.shape, dtype='float32')
        z_dummy = np.zeros(z_p.shape, dtype='float32')
        y_dummy = np.zeros((batchsize, 1), dtype='float32')
        f_dummy = np.zeros((batchsize, 8192), dtype='float32')

        # Train autoencoder
        self.enc_trainer.train_on_batch([x_r, c, z_p], [x_dummy, z_dummy])

        # Train generator
        g_loss, _, _, _, _, _, g_acc = self.dec_trainer.train_on_batch([x_r, c, z_p], [x_dummy, f_dummy, f_dummy])

        # Train classifier
        self.cls_trainer.train_on_batch([x_r, c], c_dummy)

        # Train discriminator
        d_loss, d_acc = self.dis_trainer.train_on_batch([x_r, c, z_p], y_dummy)

        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'g_acc': g_acc,
            'd_acc': d_acc
        }
        return loss

    def predict(self, z_samples):
        return self.f_dec.predict(z_samples)

    def build_model(self):
        self.f_enc = self.build_encoder(output_dims=self.z_dims*2)
        self.f_dec = self.build_decoder()
        self.f_dis = self.build_discriminator()
        self.f_cls = self.build_classifier()

        # Algorithm
        x_r = Input(shape=self.input_shape)
        c = Input(shape=(self.num_attrs,))
        z_params = self.f_enc([x_r, c]) #TODO:dim of z shold be checked

        z_avg = Lambda(lambda x: x[:, :self.z_dims], output_shape=(self.z_dims,))(z_params)
        z_log_var = Lambda(lambda x: x[:, self.z_dims:], output_shape=(self.z_dims,))(z_params) #TODO: check "x"
        z = Lambda(sample_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])

        kl_loss = KLLossLayer()([z_avg, z_log_var])

        z_p = Input(shape=(self.z_dims,))

        x_f = self.f_dec([z, c])
        x_p = self.f_dec([z_p, c])

        y_r, f_D_x_r = self.f_dis(x_r)
        y_f, f_D_x_f = self.f_dis(x_f)
        y_p, f_D_x_p = self.f_dis(x_p)

        d_loss = DiscriminatorLossLayer()([y_r, y_f, y_p])

        c_r, f_C_x_r = self.f_cls(x_r)
        c_f, f_C_x_f = self.f_cls(x_f)
        c_p, f_C_x_p = self.f_cls(x_p)

        g_loss = GeneratorLossLayer()([x_r, x_f, f_D_x_r, f_D_x_f, f_C_x_r, f_C_x_f])
        gd_loss = FeatureMatchingLayer()([f_D_x_r, f_D_x_p])
        gc_loss = FeatureMatchingLayer()([f_C_x_r, f_C_x_p])

        c_loss = ClassifierLossLayer()([c, c_r])

        # Build classifier trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, True)

        self.cls_trainer = Model(inputs=[x_r, c],
                                 outputs=[c_loss])
        self.cls_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.cls_trainer.summary()

        # Build discriminator trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, True)
        set_trainable(self.f_cls, False)

        self.dis_trainer = Model(inputs=[x_r, c, z_p],
                                 outputs=[d_loss])
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[discriminator_accuracy(y_r, y_f, y_p)])
        self.dis_trainer.summary()

        # Build generator trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, True)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, False)

        self.dec_trainer = Model(inputs=[x_r, c, z_p],
                                 outputs=[g_loss, gd_loss, gc_loss])
        self.dec_trainer.compile(loss=[zero_loss, zero_loss, zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy(y_p, y_f)])

        # Build autoencoder
        set_trainable(self.f_enc, True)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, False)

        self.enc_trainer = Model(inputs=[x_r, c, z_p],
                                outputs=[g_loss, kl_loss])
        self.enc_trainer.compile(loss=[zero_loss, zero_loss],
                                optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.enc_trainer.summary()

        # Store trainers
        self.store_to_save('cls_trainer')
        self.store_to_save('dis_trainer')
        self.store_to_save('dec_trainer')
        self.store_to_save('enc_trainer')

    def build_encoder(self, output_dims):
        """Originally network E is a GoogleNet, categorical information is mereged at the last FC layer of the E network
        """
        
        def get_VGG16():
            model_path = "./models/vgg16.h5py"
            if not os.path.exists(model_path):
                model = VGG16(weights="imagenet", include_top=False)
                model.save(model_path)
            else:
                model = load_model(model_path)
            
            for i, layer in enumerate(model.layers):
                if i < 21:
                    layer.trainable = False
                else:
                    layer.trainable = True

            return model


        x_inputs = Input(shape=self.input_shape)
        base_model = get_VGG16()

        x = base_model(x_inputs)
        x = Flatten()(x)

        c_inputs = Input(shape=(self.num_attrs,))
        x = Concatenate(axis=-1)([x, c_inputs])

        x = Dense(1024)(x)
        x = LeakyReLU(0.3)(x) 
        
        x = Dense(output_dims)(x)
        x = Activation("linear")(x)

        return Model([x_inputs, c_inputs], x)

    def build_decoder(self):
        """
        2 fully conected network followed by 6 deconv layers with 2-by-2 upsampling. The convolution layers have 256, 256, 128, 92, 64 and 3 channels with filter size of 3*3, 3*3, 5*5, 5*5, 5*5, 5*5.
        """
        z_inputs = Input(shape=(self.z_dims,))
        c_inputs = Input(shape=(self.num_attrs,))
        z = Concatenate()([z_inputs, c_inputs])

        w = self.input_shape[0] // (2 ** 4)
        x = Dense(w * w * 512)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(w * w * 512)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=1024, strides=(2, 2), kernel_size=(3,3))(x)
        x = BasicDeconvLayer(filters=512, strides=(2, 2), kernel_size=(3,3))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2), kernel_size=(5,5))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2), kernel_size=(5,5))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model([z_inputs, c_inputs], x)

    def build_discriminator(self):
        """
        use as the same network as DCGAN.
        """
        inputs = Input(shape=self.input_shape)
      
        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        #f = Flatten()(x)
        #x = Dense(1024)(f)
        f = GlobalAveragePooling2D()(x)
        x = LeakyReLU(0.3)(x)

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(inputs, [x, f])

    def build_classifier(self):
        """
        Alex net
        """
        inputs = Input(shape=self.input_shape)
      
        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)
        
        f = Flatten()(x)
        x = Dense(1024)(f)
        x = Activation('relu')(x)

        x = Dense(self.num_attrs)(x)
        #x = Activation('softmax')(x)
        x = Activation('sigmoid')(x)

        return Model(inputs, [x, f])
