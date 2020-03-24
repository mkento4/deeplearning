import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Input, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from keras.datasets import cifar10
import cv2

import keras
print(keras.__version__) 
img_cols, img_rows = 224, 224
num_classes = 10

def AlexNet():
    inputs = Input((img_cols, img_rows, 3))
    x = Conv2D(filters=96, kernel_size=(11, 11),strides=4,activation='relu', padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(5, 5),activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


if __name__ == "__main__":
    model = AlexNet()
    
    model.summary()
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    plot_model(model,to_file='alexnet_model.png',show_shapes=True)
