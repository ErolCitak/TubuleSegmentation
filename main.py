from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.layers import Dense,Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adagrad

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model


## Pre-defined parameters
batch_size = 128
num_classes = 2
epochs = 10
subtract_pixel_mean = True
data_path = "D://Biomedical_Tubule//"

model_path = "C://Users//PC//PycharmProjects//Tubule_Segmentation//"
save_dir = model_path + "Model"

network_name = "baseline"
model_name = 'alexnet_tubule_segmentation_model_b'+str(batch_size)+"_e_"+str(epochs)+"_n_"+network_name


def load_dataset(path):
    ## Load Dataset
    images = np.load(path+"images.npy")

    labels = pd.read_csv(path+"label_filename.csv")
    labels = labels.label

    # The data, split between train and test sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.33, random_state = 42)

    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    """
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train,x_test,y_train,y_test

def normalize_data(x):
    # Normalize data.
    x = x.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_mean = np.mean(x, axis=0)
        x -= x_mean

    return x


def build_alexnet_cnn(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(32, (5, 5), activation="relu", padding="same")(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(64, (5, 5), activation="relu", padding="same")(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=2)(x)

    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)

    x = Dropout(0.3)(x)
    output_layer = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


if __name__ == '__main__':

    # load dataset and normalize it
    x_train, x_test, y_train, y_test = load_dataset(data_path)
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    # build baseline, Alexnet, model
    baseline_model = build_alexnet_cnn(input_shape=(32,32,3))
    # initiate Adagrad optimizer
    opt = Adagrad(lr=0.001, decay=1e-3 * 4)

    # Let's train the model using Adagrad optimizer
    baseline_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(baseline_model.summary())

    history = baseline_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              verbose=1)

    baseline_model.save(save_dir + "//" + model_name + ".h5")

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_dir + "//" + model_name + "_acc.png")
    #plt.show()
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_dir + "//" + model_name + "_loss.png")
    #plt.show()

