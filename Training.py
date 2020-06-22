#!/usr/bin/env python
# -*-coding:utf-8 -*-
import numpy as np
from os import listdir
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


path = './' + 'features' + '/'
model_path = "./model/"
test_path = "./test_features/"

test_accuracy = []


def txtToVector(filename, N):
    returnVec = np.zeros((1,N))
    fr = open(filename)
    lineStr = fr.readline()
    lineStr = lineStr.split(' ')
    for i in range(N):
        returnVec[0, i] = int(lineStr[i])
    return returnVec

def tran_SVM(N):
    svc = SVC()
    # pre-set some parameters
    parameters = {'kernel':('linear', 'rbf'),'C':[1,3, 5, 7, 9, 11, 13, 15, 17, 19],'gamma':[0.00001,0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
    hwLabels = [] # put label
    trainingFileList = listdir(path)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,N))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = txtToVector(path+fileNameStr,N) # transform txt data into matrix format
    print("Data Loading Finished")
    clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8) # Grid search, 5-fold validation
    clf.fit(trainingMat,hwLabels)
    print(clf.return_train_score)
    print(clf.best_params_) # print the best result
    best_model = clf.best_estimator_
    print("SVM Model save...")
    save_path = model_path + "svm_efd_" + "train_model.m"
    joblib.dump(best_model,save_path)# save the best model

def test_SVM(clf,N):
    testFileList = listdir(test_path)
    errorCount = 0 # record the number of error
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNum = int(fileNameStr.split('_')[0])
        vectorTest = txtToVector(test_path+fileNameStr,N)
        valTest = clf.predict(vectorTest)
        if valTest != classNum:
            errorCount += 1
    print("error : %d times \nerror rate: %f%%" % (errorCount, errorCount/mTest * 100))

def test_fd(fd_test):
    clf = joblib.load(model_path + "svm_efd_train_model.m")
    test_svm = clf.predict(fd_test)
    return test_svm


if __name__ == "__main__":
    tran_SVM(31)
    clf = joblib.load(model_path + "svm_efd_" + "train_model.m")
    test_SVM(clf,31)



