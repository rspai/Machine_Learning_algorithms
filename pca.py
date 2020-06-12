# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
#using minmax normalization
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

#normalizing data
#from sklearn import preprocessing
#data_X = preprocessing.MinMaxScaler().fit_transform(data_X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=50)
print("After splitting into train-test")
print("x train : ", X_train.shape) 
print("x test  : ", X_test.shape) 
print("y train : ", Y_train.shape)
print("y test  : ", Y_test.shape)
print(type(Y_test))

def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    #Standardising the dataset by centering the mean and scaling each component to unit variance 
    X_train = preprocessing.scale(X_train)

    #Computing the covariance matrix to find correlation between datapoints
    X_covariance_matrix = np.cov(X_train.T)

    #Finding eigen values, eigen vectors
    eig_vals,eig_vects = np.linalg.eig(X_covariance_matrix)

    #Forming eigenvalues,eigen-vector pairs
    eig_pairs = [((eig_vals[i]),eig_vects[:,i])for i in range(len(eig_vals))]

    #Sorting the eigenvalues
    eig_pairs.sort(key=lambda X:X[0],reverse=True)

    #Getting the top n_components vectors
    n_comp_vects = eig_vects[:,:N]

    #Finding the reduced dimensionality by multiplying it with original matrix
    red_dim_mat = np.dot(X_train,n_comp_vects)
    #print(red_dim_mat)
    return red_dim_mat

red_dim_mat = PCA(X_train, 4)