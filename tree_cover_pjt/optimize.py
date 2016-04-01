##############################################
# optimize.py                                #
##############################################
# optimize some classifiers                  #
#                                            #
#                                            #
##############################################

#============================================
# import
#============================================

# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Timing
import time

#============================================
# some helpful functions
#============================================

def class_report(clf, y_true, y_pred, title=""):
# A helpful function that makes reporting the quality of the fit
# simpler.
    print("---------------------------")
    print(title)
    print("---------------------------")
    print(classification_report(y_true, y_pred))
    
    score = accuracy_score(y_pred, y_test)
    print("accuracy: %0.3f" % score)
    print("")

#=================================================
#-------------------------------------------------
#=================================================

def class_report_multi(clfs, y_true, y_pred, clf_names=0):
# Report the quality of the fit for multiple classifiers

    # If clf_names is not supplied then use a list of numbers as the names
    if clf_names==0:
        clf_names = []
        for i in range(0,len(amount_of_data)):
            clf_names.append(str(i))

    for i in range(len(clfs)):
        class_report(clfs[i], y_true, y_pred[i], title=clf_names[i])

#=================================================
#-------------------------------------------------
#=================================================

def fit_multi(clfs, x, y, x_test, clf_names=0):
# Fit multiple classifiers and return an array where each
# spot in the array is an array of the y predictions on the test set
# for that classifier

    y_pred = []
    print("")

    # If clf_names is not supplied then use a list of numbers as the names
    if clf_names==0:
        clf_names = []
        for i in range(0,len(amount_of_data)):
            clf_names.append(str(i))

    for i in range(len(clfs)):
        print("Fitting %s... " % clf_names[i])
        clf = clfs[i]
        clf.fit(x,y)
        y_pred_clf = clf.predict(x_test)
        y_pred.append(y_pred_clf)
        
    return y_pred 

#=================================================
#-------------------------------------------------
#=================================================

def optimize_parameters(clf, tuning_parameters, x_train, y_train, x_test, y_test,  scoring='accuracy', n_iter=-1):
# Optimize the parameters of the different algorithms through cross validation. Then report the results
# on the test set.

    print("")
    print("Optimizing hyper-parameters for %s ..." % scoring)
    print("")

    gs = GridSearchCV(clf, tuning_parameters, cv=5, scoring=scoring, n_jobs=-1)
    if n_iter>0:
        gs = RandomizedSearchCV(clf, tuning_parameters, cv=5, scoring=scoring, n_jobs=-1, n_iter=n_iter)
    gs.fit(x_train, y_train)

    print("")
    print("Grid scores on development set:")
    print("")
    for params, mean_score, scores in gs.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print("")
    print("Best parameters set found on development set:")
    print("")
    print(gs.best_estimator_)
    print("")
    print("Detailed classification report:")
    print("")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print("")
    y_true, y_pred = y_test, gs.predict(x_test)
    print(classification_report(y_true, y_pred))
    print("")
    print(""+scoring+": %0.3f" % gs.score(x_test, y_test))
    print("")
    return gs


#============================================
# Timing-------------------------------------
#============================================

#============================================
# load data
#============================================

# Use a different data set for training and  evaluation of goodness
df_equal = pd.read_csv('../tree_cover_pjt/data/train_multi.data', index_col=0)
#df_equal = pd.read_csv('../tree_cover_pjt/data/cover_multi.data', index_col=0)
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
# The various classifiers we want to look at
#==============================================

# --- BDT --------------------------------------
bdt = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.3, max_leaf_nodes=20, max_features=None)

# --- Random Forest ----------------------------
rf = RandomForestClassifier(n_estimators=5000, max_leaf_nodes=100, n_jobs=-1)

# --- SVC Linear -------------------------------
svc_lin = SVC(C=1, cache_size=1000)

# --- SVC RBF ----------------------------------
svc_rbf = SVC(C=100, gamma=0.001,  kernel='rbf', cache_size=1000)

# --- Logistic Regression ----------------------
lr = LogisticRegression(C=1.0)

clfs = [bdt, rf, svc_lin, svc_rbf, lr]
clf_names = ["BDT", "Random Forest", "SVC Linear", "SVC RBF", "Logistic Regression"]

#====================================================
# look at the quality of the unoptimized regression
#=====================================================

y_test_pred = fit_multi(clfs, x_train, y_train, x_test, clf_names=clf_names)
class_report_multi(clfs, y_test, y_test_pred, clf_names=clf_names)

#====================================================
# optimize the algorithms
#=====================================================
"""
# Set the parameter values to search through during optimization
# --- BDT --------------------------------------
bdt_trees = [500, 2000, 3500, 5000]
bdt_nodes = [4, 16, 32, 64]
bdt_learning_rate = [0.05, 0.1, 0.5, 1]
bdt_parameters = [{'n_estimators' : bdt_trees, 'learning_rate' : bdt_learning_rate, 'max_leaf_nodes' : bdt_nodes}]
bdt = GradientBoostingClassifier(max_features=None)
best_bdt = optimize_parameters(bdt, bdt_parameters, x_train, y_train, x_test, y_test)
"""

# --- Random Forest ----------------------------
rf_trees = [500, 1000, 2000, 5000, 10000]
rf_nodes = [16, 32, 64, 128]
rf_criterion = ['gini', 'entropy']
rf_parameters = [{'n_estimators' : rf_trees, 'criterion' : rf_criterion, 'max_leaf_nodes' : rf_nodes}]
rf = RandomForestClassifier(n_jobs=-1)
best_rf = optimize_parameters(rf, rf_parameters, x_train, y_train, x_test, y_test)

# --- SVC --------------------------------------
C=[10**(-5), 10**(-2), 1, 10**(2), 10**(5), 10**(10)]
gamma=[10**(-14), 10**(-7), 10**(-2), 1, 10**(2), 10**(4)]
for i in range(21): C.append(10.0**(i-5))
for i in range(17): gamma.append(10**(i-14))
svc_parameters = [{'kernel': ['rbf'], 'gamma': gamma, 'C': C}, {'kernel': ['linear'], 'C': C}]
svc = SVC(cache_size=1000)
best_svc = optimize_parameters(svc, svc_parameters, x_train, y_train, x_test, y_test)

# --- Logistic Regression ----------------------
lr = LogisticRegression(C=1.0)
C=[]
for i in range(21): C.append(10.0**(i-5))
lr_parameters = [{'C': C}]
best_lr = optimize_parameters(lr, lr_parameters, x_train, y_train, x_test, y_test)

