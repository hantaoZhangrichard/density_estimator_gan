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

class one_direction():
    def __init__(self):
        self.discriminator = model.Discriminator(input_dim=3, name='discriminator')
        self.generator = model.Generator(input_dim=3, output_dim=3, name='generator')
        self.sampler = util.GMM_indep_sampler(N=20000, n_components=3, dim=3)

        self.z = util.Gaussian_sampler(mean=np.zeros(3), sd=1.0)
        x = self.generator(self.z)

        optimizer = Adam(0.0002)

        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.discriminator.trainable = False

        validity = self.discriminator(x)

        self.combined = Model(self.z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, epochs, batch_size=200):


        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            z_train = self.z.train(batch_size)
            x_train = self.sampler.train(batch_size)

            gen_points = self.generator.predict(z_train)

            d_loss_real = self.discriminator.train_on_batch(x_train, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_points, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            z_train = self.z.train(batch_size)

            g_loss = self.combined.train_on_batch(z_train, valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    def sample_results(self, epoch):
        z, _ = self.z.load_all()
        gen_points = self.generator.predict(z)

        real_points, _ = self.sampler.load_all()



