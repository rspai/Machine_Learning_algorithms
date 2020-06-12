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

def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    K = N
    m = X_train.shape[0] 
    n = X_train.shape[1] 
    # Euclidean Distance Calculator
    def dist(a, b, ax=1):
        return np.linalg.norm(a - b,)

    #Taking Centroid matrix
    Centroids=np.array([]).reshape(n,0) 

    #randomly selecting centroid
    for i in range(K):
        rand = random.randint(0,m-1)
        Centroids = np.c_[Centroids,X_train[rand]]

    #Taking transpose
    centriods_use = Centroids.T

    #Storing the value of centroids when it updates
    centriods_previous = np.zeros(centriods_use.shape)

    # Calculating Error func. - Distance between new centroids and old centroids
    error = dist(centriods_use, centriods_previous, None)

    array_index = []
    centroid = []

    for i in range(len(X_train)):
        array_index.append(0)
        centroid.append(0)
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X_train)):
            length_dist = []
            for j in range(K):
                distances = dist(X_train[i], centriods_use[j])
                length_dist.append(distances)
            array_index[i] = length_dist.index(min(length_dist))
            centroid[i] = centriods_use[array_index[i]]
        centriods_previous = copy.deepcopy(centriods_use)
        # Finding the new centroids by taking the average value
        for i in range(K):
            points = [X_train[j] for j in range(len(X_train)) if array_index[j] == i]
            centriods_use[i] = np.mean(points, axis=0)  
        error = dist(centriods_use, centriods_previous, None)   
    sum1=0
    clusters=[]

    for i in range(K):
        mini_clusters = []
        for j in range(len(X_train)):
            if(array_index[j] == i):
                mini_clusters.append(X_train[j])
        clusters.append(mini_clusters)       
    clusters1 = np.array(clusters)
    return clusters1,centriods_use
    
clusters,centriods_use = Kmeans(X_train, 11)