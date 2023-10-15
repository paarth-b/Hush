import os
import re
import numpy as np
import pandas as pd
import random

import splitfolders

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    Dropout,
    Flatten,
    MaxPooling2D,
    BatchNormalization,
    Input,
    concatenate,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix

base_path = "/asl-alphabet/"

datagen = ImageDataGenerator(rescale=1.0 / 255)

train_path = "/asl-alphabet/asl_alphabet_train"
test_path = "/asl-alphabet/asl_alphabet_test"

batch = 32
image_size = 200
img_channel = 3
n_classes = 29

train_data = datagen.flow_from_directory(
    directory="asl-alphabet/asl_alphabet_train",
    target_size=(image_size, image_size),
    batch_size=batch,
    class_mode="categorical",
)

test_data = datagen.flow_from_directory(
    directory="asl-alphabet/asl_alphabet_test",
    target_size=(image_size, image_size),
    batch_size=batch,
    class_mode="categorical",
    shuffle=False,
)

# model

model = Sequential()
# input layer
# Block 1
model.add(
    Conv2D(
        32,
        3,
        activation="relu",
        padding="same",
        input_shape=(image_size, image_size, img_channel),
    )
)
model.add(Conv2D(32, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(64, 3, activation="relu", padding="same"))
model.add(Conv2D(64, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.3))

# Block 3
model.add(Conv2D(128, 3, activation="relu", padding="same"))
model.add(Conv2D(128, 3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.4))

# fully connected layer
model.add(Flatten())

model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

# output layer
model.add(Dense(36, activation="softmax"))


model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.01,
    patience=5,
    restore_best_weights=True,
    verbose=0,
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, factor=0.5, verbose=1
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# model-fitting
asl_class = model.fit(
    train_data, epochs=30, callbacks=[early_stopping, reduce_learning_rate], verbose=1
)
