# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
####k-nn algorithm from scratch
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import copy
import random
import matplotlib as plt
import sys 
sys.setrecursionlimit(10**6) 
dataframe = pd.read_csv('C:/Users/Documents/DIC/Assignment1/data.csv')
data_X = dataframe.iloc[:,0:48]
data_X = (data_X - np.min(data_X))/(np.max(data_X) - np.min(data_X)).values
data_X = np.array(data_X)
#print(data_X.shape)

data_Y = dataframe['48'].values
data_Y = np.array(data_Y)
#print(data_Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=50)
print("After splitting into train-test")
print("x train : ", X_train.shape) 
print("x test  : ", X_test.shape) 
print("y train : ", Y_train.shape)
print("y test  : ", Y_test.shape)
print(type(Y_test))

def KNN(X_train,X_test,Y_train,N):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    K = N        
    def distmat(a, b):
        return np.linalg.norm(a - b,axis=1)


    m = X_train.shape[0] 
    n = X_test.shape[0] 
    
    p = Y_train.shape[0]
    q = Y_test.shape[0]
    final_pred = []
    for i in range(n):
        arr = np.tile(X_test[i],(m,1))
        value = distmat(arr, X_train)
        sorted_indices = np.argsort(value)
        label_pred = []
        for q in range(K):
            l = sorted_indices[q]        
            label_pred.append(Y_train[l])     
        a = np.array(label_pred)
        counts = np.bincount(a) 
        final_pred.append(np.argmax(counts))
    true_pred = np.array(final_pred)
    return true_pred

y_pred = KNN(X_train, X_test, Y_train, 11)