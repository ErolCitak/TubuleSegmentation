from __future__ import print_function
import keras
from keras.layers import Input
from keras.optimizers import Adagrad,Adam

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# pretrained models
from keras.applications import vgg16, resnet50, mobilenet


## Pre-defined parameters
batch_size = 128
num_classes = 2
epochs = 15
subtract_pixel_mean = True
data_path = "D://Biomedical_Tubule//"

model_path = "C://Users//PC//PycharmProjects//Tubule_Segmentation//"
save_dir = model_path + "Model"

network_name = "baseline"
model_name = network_name+'_tubule_segmentation_model_b'+str(batch_size)+"_e_"+str(epochs)+"_n_"+network_name


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


def initialize_pretrained_models(input_shape, model_index = 0):

    if model_index == 0:

        # Load the VGG model
        vgg_model = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(32, 32, 3)))

        model_output = keras.layers.Flatten(name='flatten')(vgg_model.output)
        model_output = keras.layers.Dense(64, activation='relu', name='fc1')(model_output)
        model_output = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(model_output)

        new_model = keras.models.Model(inputs=vgg_model.input, outputs=model_output)

        for layer in vgg_model.layers:
            layer.trainable = False

    elif model_index == 1:

        # Load the Mobilenet model
        mobile_net = mobilenet.MobileNet(weights='imagenet', include_top=False, input_tensor=Input(shape=(32, 32, 3)))

        model_output = keras.layers.Flatten(name='flatten')(mobile_net.output)
        model_output = keras.layers.Dense(64, activation='relu', name='fc1')(model_output)
        model_output = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(model_output)

        new_model = keras.models.Model(inputs=mobile_net.input, outputs=model_output)

        for layer in mobile_net.layers:
            layer.trainable = False


    else:

        # Load the ResNet50 model
        resnet_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(32, 32, 3)))

        model_output = keras.layers.Flatten(name='flatten')(resnet_model.output)
        model_output = keras.layers.Dense(64, activation='relu', name='fc1')(model_output)
        model_output = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(model_output)

        new_model = keras.models.Model(inputs=resnet_model.input, outputs=model_output)

        for layer in resnet_model.layers:
            layer.trainable = False


    return new_model





if __name__ == '__main__':

    # load dataset and normalize it
    x_train, x_test, y_train, y_test = load_dataset(data_path)
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)


    for i in range(0,3):
        model = initialize_pretrained_models(input_shape=(32,32,3), model_index=i)

        if i == 0:
            model_name = "vgg16"
        elif i == 1:
            model_name = "resnet50"
        else:
            model_name = "mobilenet"

        # initiate Adagrad optimizer
        #opt = Adagrad(lr=0.001, decay=1e-3 * 4)
        opt = Adam(lr=1e-4)
        # Let's train the model using Adagrad optimizer
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        print(model.summary())

        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  verbose=1)

        model.save(save_dir + "//" + model_name +"b_"+str(batch_size)+"_e_"+str(epochs)+ ".h5")

        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(save_dir + "//" + model_name +"b_"+str(batch_size)+"_e_"+str(epochs)+ "_acc.png")
        #plt.show()
        plt.clf()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(save_dir + "//" + model_name +"b_"+str(batch_size)+"_e_"+str(epochs)+ "_loss.png")
        #plt.show()
        plt.clf()
