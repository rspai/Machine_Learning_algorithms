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
import matplotlib
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
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=50)
print("After splitting into train-test")
print("x train : ", X_train.shape) 
print("x test  : ", X_test.shape) 
print("y train : ", Y_train.shape)
print("y test  : ", Y_test.shape)
print(type(Y_test))

def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    accuracy_scores = []
    correct_pred = 0
    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i]):
            correct_pred  = correct_pred + 1
    acc = (correct_pred / float(len(y_true))) * 100.0
    accuracy_scores.append(acc)
    accuracy = sum(accuracy_scores) / len(accuracy_scores)
    return accuracy
    

def Recall(y_true,y_pred):
     """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    recall_scores = []
    true_positive_cnt = 0
    cnt = 0
    for i in range(len(y_true)):
        if(y_true[i] == 1):
            cnt = cnt + 1
            if(y_true[i] == y_pred[i]):
                true_positive_cnt = true_positive_cnt + 1 
    recall_partial = (true_positive_cnt / cnt) * 100.0
    recall_scores.append(recall_partial)
    recall = sum(recall_scores) / len(recall_scores)
    return recall

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    precision_scores = []
    true_positive_cnt = 0
    cnt = 0
    for i in range(len(y_true)):
        if(y_pred[i] == 1):
            cnt = cnt + 1
            if(y_true[i] == y_pred[i]):
                true_positive_cnt = true_positive_cnt + 1
    precision_partial = (true_positive_cnt / cnt) * 100.0
    precision_scores.append(precision_partial)
    precision = sum(precision_scores) / len(precision_scores) 
    return precision
def WCSS(Clusters,centriods_use):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    wcss = 0
    # Euclidean Distance Caculator
    def dist(a, b, ax=1):
        return np.linalg.norm(a - b)
    for i in range(len(Clusters)):
        wcss = wcss + dist(Clusters[i],centriods_use[i]) * dist(Clusters[i],centriods_use[i])
    #print(wcss)
    return wcss
def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    classes = np.unique(np.concatenate((y_true,y_pred)))
    conf_mat = np.empty((len(classes),len(classes)),dtype=np.int)
    for i,x in enumerate(classes):
        for j,y in enumerate(classes):
            conf_mat[i,j] = np.where((y_true==x) * (y_pred==y))[0].shape[0]
    
    return conf_mat

def KNN(X_train,X_test,Y_train,N):
     """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    #y_train_new=[]
    #for i in Y_train:
        #for j in i:
            #y_train_new.append(j)
    K=N        
    def distmat(a, b):
        return np.linalg.norm(a - b,axis=1)


    m=X_train.shape[0] 
    n=X_test.shape[0] 
    
    p=Y_train.shape[0]
    q=Y_test.shape[0]
    final_pred=[]
    for i in range(n):
        arr=np.tile(X_test[i],(m,1))
        value=distmat(arr,X_train)
        sorted_indices=np.argsort(value)
        label_pred=[]
        for q in range(K):
            l=sorted_indices[q]        
            label_pred.append(Y_train[l])     
        a=np.array(label_pred)
        counts = np.bincount(a) 
        final_pred.append(np.argmax(counts))
    true_pred=np.array(final_pred)
    return true_pred

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
    #X_train_data=X_train_data[0:50]

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
    
    y_pred_svm = clf.predict(X_test)


def SklearnVotingClassifier(X_train,Y_train,X_test,Y_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    #SVM (using linear kernel)
    from sklearn import svm
    clf = svm.SVC(kernel='linear') 
    clf.fit(X_train, Y_train)
    print('Accuracy for SVM: ', Accuracy(Y_test,y_pred_svm))
     
    #Logistic regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    y_pred_lr = lr.predict(X_test)
    print('Accuracy for Logistic regression: ', Accuracy(Y_test,y_pred_lr))

    #Decision Trees
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, Y_train)
    y_pred_dt = classifier.predict(X_test)
    print('Accuracy for Decision Tree: ', Accuracy(Y_test,y_pred_dt))

    #K-nn
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train, Y_train)
    y_pred_knn = knn.predict(X_test)  
    print('Accuracy for K-nn: ', Accuracy(Y_test,y_pred_knn))

    list_all = []
    list_all.append(y_pred_svm)
    list_all.append(y_pred_lr)
    list_all.append(y_pred_dt)
    list_all.append(y_pred_knn)
    return list_all



def SklearnVotingClassifier(X_train,Y_train,X_test,Y_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    import pandas
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import VotingClassifier
    
    kfold = model_selection.KFold(n_splits=5, random_state=5)

    # creating the sub models
    estimators = []
    model1 = SVC(kernel='linear')
    estimators.append(('svm', model1))
    model2 = LogisticRegression()
    estimators.append(('logistic', model2))
    model3 = DecisionTreeClassifier()
    estimators.append(('cart', model3))
    model4 = KNeighborsClassifier()
    estimators.append(('knn', model4))

    # creating the ensemble model
    ensemble_model = VotingClassifier(estimators, voting='hard')
    ensemble_model.fit(X_train, Y_train)
    y_pred_ensemble = ensemble_model.predict(X_test)
    print('Accuracy for ensemble model: ', Accuracy(Y_test,y_pred_ensemble))
    return y_pred_ensemble


"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""
def VisualizationConfusionMatrix(Y_test, y_pred):
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    labels = np.unique(np.concatenate((Y_test,y_pred))).tolist()
    cm = ConfusionMatrix(Y_test,y_pred)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()  

def GridSearchCV_hp_tuning(X_train, X_test, y_train, y_test):
    ##for SVM for best parameters
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV 
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    param_grid = {'C': [1, 10],  
                  'gamma': ('auto','scale'), 
                  'kernel': ['linear']}  
    grid = GridSearchCV(SVC(), param_grid, cv=2) 
    grid.fit(X_train, y_train)
    print('Best parameters: ', grid.best_params_) 
    print('Best estimator: ', grid.best_estimator_) 
    grid_predictions = grid.predict(X_test)
    acc = Accuracy(grid_predictions, y_test)
    print('Acc: ', acc)
    print(classification_report(y_test, grid_predictions))
    
    ##for Knn for best parameters
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    params_knn = {'n_neighbors': [2,3,4,5], 
                  'weights': ['uniform'], 
                  'metric': ['euclidean'] 
                 }
    knn_grid = GridSearchCV(knn, params_knn, cv=3)
    knn_grid.fit(X_train, y_train)
    print('Best parameters: ', knn_grid.best_params_) 
    print('Best estimator: ', knn_grid.best_estimator_) 
    grid_predictions = knn_grid.predict(X_test)
    acc = Accuracy(grid_predictions, y_test)
    print('Acc: ', acc)
    print(classification_report(y_test, grid_predictions))
    
    ##for Decision Tree for best parameters
    from sklearn.tree import DecisionTreeClassifier as dt
    clf = dt()
    param_grid = {'max_depth':[1,2,3],
                  'min_samples_leaf':[1,2,3,4,5],
                  'min_samples_split':[2,3,4],
                  'criterion':['gini','entropy']
                 }
    grid = GridSearchCV(clf, param_grid, cv=10)
    grid.fit(X_train, y_train)
    print('Best parameters: ', grid.best_params_) 
    print('Best estimator: ', grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    acc = Accuracy(grid_predictions, y_test)
    print('Acc: ', acc)
    print(classification_report(y_test, grid_predictions))
    
    ##Tuning the hyperparameters
    ##for SVM: parameter C
    from sklearn.model_selection import GridSearchCV 
    from sklearn.metrics import classification_report 
    c_values = [0.1, 1, 10 , 100]
    acc = []
    for i in c_values:
        param_grid = {'C': [i],  
                      'gamma': ('auto','scale'), 
                      'kernel': ['linear']}  
        grid = GridSearchCV(SVC(), param_grid, cv=2) 
        grid.fit(X_train, y_train)
        print('Best parameters: ', grid.best_params_) 
        print('Best estimator: ', grid.best_estimator_) 
        grid_predictions = grid.predict(X_test)
        acc_1 = Accuracy(grid_predictions, y_test)
        acc.append(acc_1) 
    xi = list(range(len(c_values)))
    plt.plot(xi, acc, marker='o', linestyle='--', color='r', label='acc')
    plt.xlabel('C values',fontweight="bold",fontsize = 12)
    plt.ylabel('accuracy',fontweight="bold",fontsize = 12)
    plt.title("C vs accuracy for GridSearchCV SVM",fontweight="bold",fontsize = 16)
    plt.xticks(xi, c_values)
    plt.legend()
    plt.show()
    
    ##for SVM: parameter kernel
    from sklearn.model_selection import GridSearchCV 
    from sklearn.metrics import classification_report 
    kernel_values = ['linear', 'rbf']
    acc_k = []
    for i in kernel_values:
        # defining parameter range 
        param_grid = {'C': [10],  
                      'gamma': ('auto','scale'), 
                      'kernel': [i]}  
        grid = GridSearchCV(SVC(), param_grid, cv=2) 
        grid.fit(X_train, y_train)
        print('Best parameters: ', grid.best_params_) 
        print('Best estimator: ', grid.best_estimator_) 
        grid_predictions = grid.predict(X_test)
        acc_1 = Accuracy(grid_predictions, y_test)
        acc_k.append(acc_1)
    xi = list(range(len(kernel_values)))
    plt.plot(xi, acc_k, marker='o', linestyle='--', color='r', label='acc')
    plt.xlabel('kernel',fontweight="bold",fontsize = 12)
    plt.ylabel('accuracy',fontweight="bold",fontsize = 12)
    plt.title("kernels vs accuracy for GridSearchCV SVM",fontweight="bold",fontsize = 16)
    plt.xticks(xi, kernel_values)
    plt.legend()
    plt.show()

    ##for decision tree: parameter max_depth
    from sklearn.tree import DecisionTreeClassifier as dt
    max_depth_values = [1, 2, 3]
    acc_dep = []
    clf=dt()
    for i in max_depth_values:
        param_grid = {'max_depth':[i],
                      'min_samples_leaf':[1,2,3,4,5],
                      'min_samples_split':[2,3,4],
                      'criterion':['gini','entropy']}
        grid = GridSearchCV(clf,param_grid, cv=10)
        a = grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        print('Best parameters: ', grid.best_params_) 
        print('Best estimator: ', grid.best_estimator_)  
        grid_predictions = grid.predict(X_test)
        acc = Accuracy(grid_predictions, y_test)
        acc_dep.append(acc)
    xi = list(range(len(max_depth_values)))
    plt.plot(xi, acc_dep, marker='o', linestyle='--', color='r', label='acc')
    plt.xlabel('max_depth values',fontweight="bold",fontsize = 12)
    plt.ylabel('accuracy',fontweight="bold",fontsize = 12)
    plt.title("max_depth vs accuracy for GridSearchCV Decision Tree",fontweight="bold",fontsize = 16)
    plt.xticks(xi, max_depth_values)
    plt.legend()
    plt.show()
    
    ##for Knn: parameter K
    knn = KNeighborsClassifier()
    acc_knn = []
    n_values = [2, 3, 4, 5]
    for i in n_values:
        params_knn = {'n_neighbors': [i], 
                      'weights': ['uniform'], 
                      'metric': ['euclidean'] 
                     }
        knn_grid= GridSearchCV(knn, params_knn, cv=3)
        knn_grid.fit(X_train, y_train)
        print('Best parameters: ', grid.best_params_) 
        print('Best estimator: ', grid.best_estimator_)  
        grid_predictions = knn_grid.predict(X_test)
        acc_1 = Accuracy(grid_predictions, y_test)
        print('acc: ',acc_1)
        acc_knn.append(acc_1)
    xi = list(range(len(n_values)))
    plt.plot(xi, acc_knn, marker='o', linestyle='--', color='r', label='acc')
    plt.xlabel('k values',fontweight="bold",fontsize = 12)
    plt.ylabel('accuracy',fontweight="bold",fontsize = 12)
    plt.title("k vs accuracy for GridSearchCV Knn",fontweight="bold",fontsize = 16)
    plt.xticks(xi, n_values)
    plt.legend()
    plt.show()


    
