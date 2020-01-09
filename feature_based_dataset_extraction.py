import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage.feature.texture import local_binary_pattern

feature_list = ["HOG","HIST","GLCM"]

## Pre-defined parameters
num_classes = 2

glcm_patch_size = 32
data_path = "D://Biomedical_Tubule//size_"+str(glcm_patch_size)+"//"


model_path = "C://Users//PC//PycharmProjects//Tubule_Segmentation//Features"
save_dir = model_path + "//"+str(glcm_patch_size)

feature_name = ""


def load_dataset(path):
    ## Load Dataset
    images = np.load(path+"images.npy")

    labels = pd.read_csv(path+"label_filename.csv")
    labels = labels.label
    labels = labels[:images.shape[0]]

    # The data, split between train and test sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.33, random_state = 42)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    y_train_num = np.zeros(y_train.shape[:1])
    y_test_num = np.zeros(y_test.shape[:1])

    for i in range(len(y_train)):

        if (y_train[i] == [0,1]).all():
            y_train_num[i] = 1
        else:
            y_train_num[i] = 0

    for i in range(len(y_test)):

        if (y_test_num[i] == [0,1]).all():
            y_test_num[i] = 1
        else:
            y_test_num[i] = 0


    return x_train,x_test,y_train_num.astype(int),y_test_num.astype(int)

def extract_feature(data, label, train_test = 1):

    # for example; data.shape = (243699, 64, 64, 3)
    hog_features = []
    hist_features = []
    glcm_features = []

    hog_labels = []
    hist_labels = []

    # extract each feature set from same data
    for feature_name in (feature_list):

        if feature_name == "HOG":
            print("Hog is starting")
            hog = cv2.HOGDescriptor()

            for i in range(data.shape[0]):
                # get image
                image = data[i]

                # convert it into grayscale
                gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_im = cv2.resize(gray_im, (64, 128), interpolation=cv2.INTER_AREA)
                feature_im = hog.compute(gray_im)

                # print(feature_im.shape) = 3780 x 1
                hog_features.append(feature_im)
                hog_labels.append(label[i])

        elif feature_name == "HIST":
            print("Histogram is starting")

            for i in range(data.shape[0]):
                # get image
                image = data[i]

                bgr_planes = cv2.split(image)

                histSize = 256
                histRange = (0, 256)  # the upper boundary is exclusive
                accumulate = False

                b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
                g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
                r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

                ## print(np.concatenate((b_hist,g_hist,r_hist), axis=0).shape)  == 768 x 1
                feature_vec = np.concatenate((b_hist,g_hist,r_hist), axis=0)
                hist_features.append(feature_vec)
                hist_labels.append(label[i])

        elif feature_name == "GLCM_XX":
            print("GLCM Starting")
            for i in range(data.shape[0]):
                # get image
                image = data[i]

                # convert it into grayscale
                gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                glcm = greycomatrix(gray_im, [5], [0], 256, symmetric=True, normed=True).flatten()

                ## print(glcm.flatten().shape) = 65536 x 1

                glcm_features.append(glcm)


    # now save the features
    save_status = False

    indicator = "Train"
    if train_test == 1:
        indicator = "Train"
    else:
        indicator = "Test"

    try:
        np.save( save_dir+"//hog_"+indicator+'.npy', np.array(hog_features))
        np.save(save_dir + "//hist_" + indicator + '.npy', np.array(hist_features))

        np.save( save_dir+"//hog_label_"+indicator+'.npy', np.array(hog_labels))
        np.save(save_dir + "//hist_label_" + indicator + '.npy', np.array(hist_labels))

        #np.save(save_dir + "//glcm_" + indicator + '.npy', np.array(glcm_features))

        save_status = True
    except:
        save_status = False


    return save_status

if __name__ == '__main__':

    # load dataset and normalize it
    x_train, x_test, y_train, y_test = load_dataset(data_path)

    # extract features from train and test
    print("Train's starting")
    status_train = extract_feature(x_train, y_train, 1)

    print("Test's starting")
    status_test = extract_feature(x_test, y_test, 0)

    if status_train == True:
        if status_test == True:
            print("Feature Extraction is Success!!")
        else:
            print("Test is NOT Success!!")
    else:
        print("Train is NOT Success!!")
