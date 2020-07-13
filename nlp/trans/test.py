__author__ = 'jeff'

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tensorflow import keras

# layer = keras.layers.Dense(3)
# layer.build((None, 4))  # Create the weights
#
# print("weights:", len(layer.weights))
# print("trainable_weights:", len(layer.trainable_weights))
# print("non_trainable_weights:", len(layer.non_trainable_weights))


def trans_test():
    base_model = keras.applications.Xception(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False)  # Do no
    base_model.trainable = False

    inputs = keras.Input(shape=(150, 150, 3))
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])
    model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)

    base_model.trainable = True

    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])

    model.fit(new_dataset, epochs=10, callbacks=..., validation_data=...)

