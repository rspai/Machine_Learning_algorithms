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

def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    import sys 
    sys.setrecursionlimit(10**6) 
    n_trees=5
    n_bootstrap=100
    n_feat=2
    y_train = Y_train.reshape(Y_train.shape[0], 1)
   
    X_train_data=np.concatenate((X_train,y_train),axis=1)
    X_train_data=X_train_data[0:50]

    #def random_forest(X_train_data,n_trees,n_bootstrap,n_feat):
    forest=[]
    
    
    
    def get_col_based_splits(X_train_data,col_split,val_split):
        split_col_vals=X_train_data[:,col_split]
        greater_vals=X_train_data[split_col_vals > val_split]
        smaller_vals=X_train_data[split_col_vals <= val_split]
        return smaller_vals,greater_vals

    def compute_entropy(X_train_data):
        label=X_train_data[:,-1]
        _,cnt=np.unique(label,return_counts=True) 
        probabilities=cnt/cnt.sum()
        entropy=sum(probabilities*-np.log2(probabilities)) #elementwise probability 
        return entropy

    def bootstrapping(X_train_data,n_bootstrap):
        bootstrap_indices = np.random.randint(low=0, high=len(X_train_data), size=n_bootstrap)
        df_bootstrapped = X_train_data[bootstrap_indices]
        return df_bootstrapped

    def decision_tree(X_train_data,flag=0,min_samples=2,max_depth=5):
        label=X_train_data[:,-1]
        a=np.unique(label)
        unique_classes, Uniq_counts=np.unique(label,return_counts=True)
        largest_idx=Uniq_counts.argmax()
        classification=unique_classes[largest_idx]
        if len(a)==1:
            return True
        else:
            flag+=1
            n_splits={}
            _,cols=X_train_data.shape
            for i in range (cols-1):
                n_splits[i]=[]
                vals=np.unique(X_train_data[:,i])

                for j in range(len(vals)):
                    if j!=0:
                        curr_val=vals[j]
                        prev_val=vals[j-1]
                        split=(curr_val+prev_val)/2
                        n_splits[i].append(split)
            overall_entropy=999
            for i in n_splits:
                for val in n_splits[i]:
                    smaller_vals,greater_vals=get_col_based_splits(X_train_data,col_split=i,val_split=val)
                    all_data=len(smaller_vals)+len(greater_vals)
                    smaller_vals_pts=len(smaller_vals)/all_data
                    greater_vals_pts=len(greater_vals)/all_data
                    curr_total_entropy=(smaller_vals_pts*compute_entropy(smaller_vals)+greater_vals_pts*compute_entropy(greater_vals))
                if curr_total_entropy<=overall_entropy:
                    overall_entropy=curr_total_entropy
                    best_split_col=i
                    best_split_val=val
        #n_splits=get_nsplits(X_train)
        #col_split,val_split=best_split(X_train,n_splits)
        #print(col_split,val_split)
        smaller_vals,greater_vals=get_col_based_splits(X_train_data,best_split_col,best_split_val)
        quest="{} <= {}".format(best_split_col,best_split_val)
        subtree={quest:[]}
        ans_y=decision_tree(smaller_vals,flag,min_samples,max_depth)
        ans_n=decision_tree(greater_vals,flag,min_samples,max_depth)
        subtree[quest].append(ans_y)
        subtree[quest].append(ans_n)

        return subtree

    for x in range (n_trees):
        df_bootstrap=bootstrapping(X_train_data,n_bootstrap)
        tree=decision_tree(df_bootstrap)
        forest.append(tree)
    forest=np.array(forest)
    
    return forest

rand = RandomForest(X_train,Y_train,X_test)