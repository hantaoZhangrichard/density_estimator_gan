import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import LeakyReLU


class Discriminator(object):
    def __init__(self, input_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self):
        model = Sequential()
        for _ in range(self.nb_layers):
            model.add(Dense(self.nb_units))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
        model.add(Dense(1, activation='sigmoid'))
        model.add(layers.BatchNormalization)
        points = Input(shape=(self.input_dim, ))
        output = model(points)
        return Model(points, output)


class Generator(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self):
        model = Sequential()
        for _ in range(self.nb_layers):
            model.add(Dense(self.nb_units))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
        model.add(Dense(self.output_dim, activation='tanh'))
        noise = Input(shape=(self.input_dim, ))
        points = model(noise)
        return Model(noise, points)


