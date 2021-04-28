import argparse

import numpy as np
from os import makedirs
from os.path import exists
import tensorflow as tf

CLIP_MIN = -0.5
CLIP_MAX = 0.5

K = tf.keras.backend
mnist = tf.keras.datasets.mnist
np_utils = tf.keras.utils
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
l2 = tf.keras.regularizers.l2


def train():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    layers = [
        Conv2D(64, (3, 3), padding="valid", input_shape=(28, 28, 1)),
        Activation("relu"),
        Conv2D(64, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128),
        Activation("relu"),
        Dropout(0.5),
        Dense(10),
    ]


    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.add(Activation("softmax"))

    print(model.summary())
    model.compile(
        loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=128,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    if not exists("model"):
        makedirs("model")

    model.save("./models/model_{}.h5".format("mnist"))


if __name__ == "__main__":

    train()
