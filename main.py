import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import LeakyReLU
import util
import model

'''
Instructions: Roundtrip model for density estimation
    x,y - data drawn from base density and observation data (target density)
    y_  - learned distribution by G(.), namely y_=G(x)
    x_  - learned distribution by H(.), namely x_=H(y)
    y__ - reconstructed distribution, y__ = G(H(y))
    x__ - reconstructed distribution, x__ = H(G(y))
    G(.)  - generator network for mapping x space to y space
    H(.)  - generator network for mapping y space to x space
    Dx(.) - discriminator network in x space (latent space)
    Dy(.) - discriminator network in y space (observation space)
'''


class RoundTrip():
    def __init__(self, g_net, h_net, dx_net, dy_net, x_sampler, y_sampler, data, pool, batch_size, alpha, beta, df,
                 is_train):
        self.data = data
        self.g_net = g_net
        self.h_net = h_net
        self.dx_net = dx_net
        self.dy_net = dy_net
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.df = df
        self.pool = pool
        self.x_dim = self.dx_net.input_dim
        self.y_dim = self.dy_net.input_dim

        self.x = tf.Tensor(shape=[None, self.dx_net.input_dim], name='x')
        self.y = tf.Tensor(shape=[None, self.dy_net.input_dim], name='y')

        self.y_ = self.g_net(self.x)
        # self.J = batch_jacobian(self.y_, self.x)
        self.x_ = self.h_net(self.y)

        self.dy_ = self.dy_net(self.y_)
        self.dx_ = self.dx_net(self.x_)

        self.x__ = self.h_net(self.y_)
        self.y__ = self.g_net(self.x_)

        self.l2_loss_x = tf.keras.losses.MSE(self.x, self.x__)
        self.l2_loss_y = tf.keras.losses.MSE(self.y, self.y__)

        self.g_loss_adv = tf.keras.losses.binary_crossentropy(tf.ones_like(self.dy_), self.dy_)
        self.h_loss_adv = tf.keras.losses.binary_crossentropy(tf.ones_like(self.dx_), self.dx_)

        self.g_loss = self.g_loss_adv + self.alpha * self.l2_loss_x + self.beta * self.l2_loss_y
        self.h_loss = self.h_loss_adv + self.alpha * self.l2_loss_x + self.beta * self.l2_loss_y
        self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha * self.l2_loss_x + self.beta * self.l2_loss_y

        self.dx = self.dx_net(self.x)
        self.dy = self.dy_net(self.y)

        self.fake_x = tf.Tensor(shape=[None, self.x_dim], name='fake_x')
        self.fake_y = tf.Tensor(shape=[None, self.y_dim], name='fake_y')

        self.d_fake_x = self.dx_net(self.fake_x)
        self.d_fake_y = self.dy_net(self.fake_y)

        self.dx_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(self.dx), self.dx) \
                       + tf.keras.losses.binary_crossentropy(tf.zeros_like(self.d_fake_x), self.d_fake_x)
        self.dy_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(self.dy), self.dy) \
                       + tf.keras.losses.binary_crossentropy(tf.zeros_like(self.d_fake_y), self.d_fake_y)
        self.d_loss = self.dx_loss + self.dy_loss


