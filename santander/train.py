##############################################
# train.py                                   #
##############################################
# train some classifiers, then optimize them #
#                                            #
#                                            #
##############################################

#============================================
# import
#============================================

import sys 
sys.path.insert(0, '../lib')

# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algo_error_and_timing import algos_error_vs_data

# Sci kit learn preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

# Sci kit learn fitting
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Sci kit learn evaluation
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Timing
import time

#============================================
# load data
#============================================

# Get the data set for algo timing info (this one has more events but unequal #'s of y values in each class) 
df_equal = pd.read_csv('data/train.csv', index_col=0)
#df_equal = pd.read_csv('../tree_cover_pjt/data/train_multi.data', index_col=0)
#df_equal = df_equal.reindex(np.random.permutation(df_equal.index))

#==============================================
# organize data into testing and training sets
#==============================================

# get a vector of the y values
y = df_equal['TARGET']
# get a matrix of the x values
x = df_equal.ix[:, df_equal.columns != 'TARGET']

# Standardize the features
x = scale(x)
print("")
print("---------------------------")
print("y counts")
print("---------------------------")
print(y.value_counts().sort_index())
print("")

weights = {0:0.04, 1:0.96}

# split into a training and testing set where the testing set is 40% of the total
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2, random_state=0)

# The various classifiers we want to look at
bdt = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.3, max_leaf_nodes=20, max_features=None) 
#rf = RandomForestClassifier(n_estimators=5000, max_leaf_nodes=100, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=5000, max_leaf_nodes=100, n_jobs=-1, class_weight=weights)
#svc_lin = SVC(C=1, cache_size=1000, probability=True)
svc_lin = SVC(C=1, cache_size=1000, class_weight=weights, probability=True)
#svc_rbf = SVC(C=100, gamma=0.001,  kernel='rbf', probability=True, cache_size=1000)
svc_rbf = SVC(C=100, gamma=0.001, class_weight=weights, kernel='rbf', probability=True, cache_size=1000)
#lr = LogisticRegression()
lr = LogisticRegression(C=1.0, class_weight=weights)


clfs = [rf, svc_lin, svc_rbf, lr, bdt]
clf_names = ["Random Forest 5000 Trees", "Linear SVC", "RBF_SVC", "Logistic_Regression", "BDT 1000 Trees 20 Nodes"]

# Timing and error info
amount_of_data, times, scores = algos_error_vs_data(clfs, x_train, y_train, x_test, y_test, percent_of_data=[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5], clf_names=clf_names, score_type='roc', timelimit=100)

print amount_of_data
print times
print scores

print "Random Forest Feature Importance..."
print rf.feature_importances_

plot_timing_and_error(amount_of_data, times, scores, score_type='roc', clf_names=clf_names)
plt.show()

"""
#============================================
# Training and Evaluation--------------------
#============================================

# Use a different data set for training and  evaluation of goodness
df_equal = pd.read_csv('../tree_cover_pjt/data/train_multi.data', index_col=0)
df_equal = df_equal.reindex(np.random.permutation(df_equal.index))

# get the column of the y values
y_loc = df_equal.shape[1] -1
# get a vector of the y values
y = df_equal.ix[:,y_loc]
# get a matrix of the x values
x = df_equal.ix[:,0:y_loc]

# Standardize the features
x = scale(x)
print("")
print("---------------------------")
print("y counts")
print("---------------------------")
print(y.value_counts().sort_index())
print("")

# split into a training and testing set where the testing set is 40% of the total
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.4, random_state=0)

#==============================================
# train the regressor/classifier 
#==============================================


y_true, y_pred_rf = y_test, rf.predict(x_test)

svc = SVC(C=1, cache_size=1000)
svc.fit(x_train,y_train)

y_pred_svc = svc.predict(x_test)

#==============================================
# look at the quality of the regression
#==============================================

class_report(rf, y_true, y_pred_rf, title="Unoptimized Random Forest")
class_report(svc, y_true, y_pred_svc, title="Unoptimized SVC")

# Set the parameter values to search through during optimization

C=[1, 10, 100, 1000]
gamma=[1e-3, 1e-4]
#Better Search space that will take longer
#for i in range(21): C.append(10.0**(i-5))
#for i in range(17): gamma.append(10**(i-14))

svc_parameters = [{'kernel': ['rbf'], 'gamma': gamma,
                     'C': C},
                    {'kernel': ['linear'], 'C': C}]

print("Optimized SVC")
print("---------------------------")

svc = SVC(C=1, cache_size=1000)
gs_svc = optimize_parameters(svc, svc_parameters, x_train, y_train, x_test, y_test)
#gs_rf = optimize_parameters(rf, rf_parameters)
"""
