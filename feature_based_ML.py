import numpy as np
import os
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import pickle

# Some constants
features_main_path = "C://Users//PC//PycharmProjects//Tubule_Segmentation//Features"
model_save_path = "C://Users//PC//PycharmProjects//Tubule_Segmentation//Traditional"
model_output_path = "C://Users//PC//PycharmProjects//Tubule_Segmentation//Traditional_Result"
size_ = ["32","64"]
n_component = 500

hog_ = ["hog_Train.npy","hog_Test.npy"]
hist_ = ["hist_Train.npy","hist_Test.npy"]

hog_label_ = ["hog_label_Train.npy","hog_label_Test.npy"]
hist_label_ = ["hist_label_Train.npy","hist_label_Test.npy"]

def classification_report_csv(report, save_name):

    """
    df = pd.DataFrame(report)
    df.to_csv(save_name, index = ['acc','precision','recall','f1'])
    """
    np.save(save_name, np.array(report))

def svm_train(data, label):

    print("SVM Training")
    clf = LinearSVC(random_state=0, tol=1e-5)

    clf.fit(data,label)

    return clf

def dt_train(data, label):

    print("DT Training")
    clf = tree.DecisionTreeClassifier()

    clf.fit(data,label)

    return clf


def rf_train(data, label):


    print("RF Training")
    clf = RandomForestClassifier(max_depth=5, random_state=0)

    clf.fit(data,label)

    return clf

def log_reg_train(data, label):

    print("LR Training")
    clf = LogisticRegression(random_state=0)

    clf.fit(data,label)

    return clf

def create_classification_result(model ,data, label):

    # model output
    y_pred = model.predict(data)

    acc = accuracy_score(label, y_pred)
    pre = precision_score(label, y_pred)
    recall = recall_score(label, y_pred)
    f1 = f1_score(label, y_pred, average='weighted')

    clf_result = [acc, pre, recall, f1]

    return clf_result


if __name__=="__main__":


    for size in size_:
        size = str(size)
        ###
        hog_train = np.squeeze(np.load(features_main_path+"//"+size+"//"+hog_[0]), axis=2)
        hog_train_label = np.load(features_main_path+"//"+size+"//"+hog_label_[0])

        hog_test = np.squeeze(np.load(features_main_path+"//"+size+"//"+hog_[1]), axis=2)
        hog_test_label = np.load(features_main_path+"//"+size+"//"+hog_label_[1])

        """
        pca = PCA(n_components = n_component)
        pca_hog = pca.fit(np.vstack((hog_train,hog_test)))

        hog_train = pca_hog.transform(hog_train)
        hog_test = pca_hog.transform(hog_test)
        """

        #####################################################################################

        hist_train = np.squeeze(np.load(features_main_path+"//"+size+"//"+hist_[0]), axis=2)
        hist_train_label = np.load(features_main_path+"//"+size+"//"+hist_label_[0])

        hist_test = np.squeeze(np.load(features_main_path+"//"+size+"//"+hist_[1]), axis=2)
        hist_test_label = np.load(features_main_path+"//"+size+"//"+hist_label_[1])

        """
        pca = PCA(n_components = n_component)
        pca_hist = pca.fit(np.vstack((hist_train,hist_test)))

        hist_train = pca_hist.transform(hist_train)
        hist_test = pca_hist.transform(hist_test)
        """


        ###
            ### TRAIN
        ###

        ## svm ##
        svm_model_hog = svm_train(hog_train, hog_train_label)
        svm_model_hist = svm_train(hist_train, hist_train_label)

        ### save svm models
        filename = model_save_path+'//'+str(size)+'//'+'svm_hog_'+str(size)+'.sav'
        pickle.dump(svm_model_hog, open(filename, 'wb'))

        filename = model_save_path+'//'+str(size)+'//'+'svm_hist_'+str(size)+'.sav'
        pickle.dump(svm_model_hist, open(filename, 'wb'))

        ## decision_tree ##
        dt_model_hog = dt_train(hog_train, hog_train_label)
        dt_model_hist = dt_train(hist_train, hist_train_label)

        ### save dt models
        filename = model_save_path+'//'+str(size)+'//'+'dt_hog_' + str(size) + '.sav'
        pickle.dump(dt_model_hog, open(filename, 'wb'))

        filename = model_save_path+'//'+str(size)+'//'+'dt_hist_' + str(size) + '.sav'
        pickle.dump(dt_model_hist, open(filename, 'wb'))

        ## random_forest ##
        rf_model_hog = rf_train(hog_train, hog_train_label)
        rf_model_hist = rf_train(hist_train, hist_train_label)

        ### save rf models
        filename = model_save_path+'//'+str(size)+'//'+'rf_hog_' + str(size) + '.sav'
        pickle.dump(rf_model_hog, open(filename, 'wb'))

        filename = model_save_path+'//'+str(size)+'//'+'rf_hist_' + str(size) + '.sav'
        pickle.dump(rf_model_hist, open(filename, 'wb'))


        ## log_reg ##
        lr_model_hog = log_reg_train(hog_train, hog_train_label)
        lr_model_hist = log_reg_train(hist_train, hist_train_label)

        ## save log_reg models
        filename = model_save_path+'//'+str(size)+'//'+'lr_hog_' + str(size) + '.sav'
        pickle.dump(lr_model_hog, open(filename, 'wb'))

        filename = model_save_path+'//'+str(size)+'//'+'lr_hist_' + str(size) + '.sav'
        pickle.dump(lr_model_hist, open(filename, 'wb'))


        ###
            ### TEST
        ###
        print("Test svm")
        svm_model_hog_res = create_classification_result(svm_model_hog, hog_test, hog_test_label)
        svm_model_hist_res = create_classification_result(svm_model_hist, hist_test, hist_test_label)

        print("Test dt")
        dt_model_hog_res = create_classification_result(dt_model_hog, hog_test, hog_test_label)
        dt_model_hist_res = create_classification_result(dt_model_hist, hist_test, hist_test_label)

        print("Test rf")
        rf_model_hog_res = create_classification_result(rf_model_hog, hog_test, hog_test_label)
        rf_model_hist_res = create_classification_result(rf_model_hist, hist_test, hist_test_label)

        print("Test lr")
        lr_model_hog_res = create_classification_result(lr_model_hog, hog_test, hog_test_label)
        lr_model_hist_res = create_classification_result(lr_model_hist, hist_test, hist_test_label)

        ## save
        filename = model_output_path+'//'+str(size)+'//'+'svm_hog_res_'+str(size)+'.npy'
        classification_report_csv(svm_model_hog_res, filename)

        filename = model_output_path+'//'+str(size)+'//'+'svm_hist_res_'+str(size)+'.npy'
        classification_report_csv(svm_model_hist_res, filename)

        ## save
        filename = model_output_path+'//'+str(size)+'//'+'dt_hog_res_' + str(size) + '.npy'
        classification_report_csv(dt_model_hog_res, filename)

        filename = model_output_path+'//'+str(size)+'//'+'dt_hist_res_' + str(size) + '.npy'
        classification_report_csv(dt_model_hist_res, filename)

        ## save
        filename = model_output_path+'//'+str(size)+'//'+'rf_hog_res_' + str(size) + '.npy'
        classification_report_csv(rf_model_hog_res, filename)

        filename = model_output_path+'//'+str(size)+'//'+'rf_hist_res_' + str(size) + '.npy'
        classification_report_csv(rf_model_hist_res, filename)

        ## save
        filename = model_output_path+'//'+str(size)+'//'+'lr_hog_res_' + str(size) + '.npy'
        classification_report_csv(lr_model_hog_res, filename)

        filename = model_output_path+'//'+str(size)+'//'+'lr_hist_res_' + str(size) + '.npy'
        classification_report_csv(lr_model_hist_res, filename)
