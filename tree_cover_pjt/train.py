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
    
    score = clf.score(x_test, y_test)
    print("accuracy: %0.3f" % score)
    print("")

#=================================================
#-------------------------------------------------
#=================================================

def optimize_parameters(clf, tuning_parameters, x_train, y_train, x_test, y_test,  scoring='accuracy'):
# Optimize the parameters of the different algorithms through cross validation. Then report the results
# on the test set.

    print("")
    print("Optimizing hyper-parameters for %s ..." % scoring)
    print("")

    gs = GridSearchCV(clf, tuning_parameters, cv=5, scoring=scoring)
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

#=================================================
#-------------------------------------------------
#=================================================

def time_algo_vs_data(clf, x, y, clf_name="", percent_of_data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], timelimit=-1):
# Plot the time it takes to run the algorithm vs the amount of data
    print("---------------------------")
    print("Timing %s ..." % clf_name)
    print("---------------------------")

    # We can plot the time it takes vs the amount of data or vs the percent of total data
    amt_of_data = []
    # The total amount of data to work with
    total_data = x.shape[0]
    # This will contain the list of times that we will plot vs amt_of_data or percent_of_data
    runtimes = []

    # for each percent of the total amount of data specified, calculate the time it takes to run the algo
    for percent in percent_of_data:
        # Absolute measure of the data used, calculated from the percent of data used
        amt_to_use = int(total_data*percent)
        amt_of_data.append(amt_to_use)
        # Run the algorithm on the amount of data specified and see how long it takes
        x_subset,y_subset = x[:amt_to_use], y[:amt_to_use]
        start = time.time()
        clf.fit(x_subset,y_subset)
        end = time.time()
        runtime = (end-start)
        # Put how long it took into a vector so that we can plot it later
        runtimes.append(runtime)
        print("%0.003f , %0.3f s" % (percent, runtime))
        if timelimit!=-1: 
            if runtime > timelimit:
                break


    print("")
    timing_plot = plt.plot(amt_of_data, runtimes, label=clf_name)
    plt.xlabel('Amount of Training Data')
    plt.ylabel('Time in Seconds')
    plt.title(clf_name)
    return timing_plot

#=================================================
#-------------------------------------------------
#=================================================

def time_algos_vs_data(clfs, x, y, clf_names=0, percent_of_data = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], timelimit=-1):
    # If names is not supplied then use a list of numbers as the names
    if clf_names==0:
        clf_names = []
        for i in range(0,len(clfs)):
            clf_names.append(str(i))

    # Plot all of the graphs for the different algorithms
    for i in range(0,len(clfs)):
        time_algo_vs_data(clfs[i], x, y, clf_names[i], percent_of_data=percent_of_data, timelimit=timelimit)

    # Set the attributes of the plot
    plt.legend(loc="upper left")
    plt.title("Timing for Various Algorithms")

#=================================================
#-------------------------------------------------
#=================================================

def algo_error_vs_data(clf, x, y, x_test, y_test, clf_name="", percent_of_data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], timelimit=-1):
# Plot the time it takes to run the algorithm vs the amount of data
    print("---------------------------")
    print("Timing %s ..." % clf_name)
    print("---------------------------")

    # We can plot the time it takes vs the amount of data or vs the percent of total data
    amt_of_data = []
    # The total amount of data to work with
    total_data = x.shape[0]
    # This will contain the list of times that we will plot vs amt_of_data or percent_of_data
    runtimes = []
    scores = []

    # for each percent of the total amount of data specified, calculate the time it takes to run the algo
    for percent in percent_of_data:
        # Absolute measure of the data used, calculated from the percent of data used
        amt_to_use = int(total_data*percent)
        amt_of_data.append(amt_to_use)
        # Run the algorithm on the amount of data specified and see how long it takes
        x_subset,y_subset = x[:amt_to_use], y[:amt_to_use]
        start = time.time()
        clf.fit(x_subset,y_subset)
        end = time.time()
        runtime = (end-start)
        score = accuracy_score(clf.predict(x_test), y_test)
        #score = f1_score(x_test, y_test, average='macro', 'micro', or None)

        # Put the goodness of fit score into a vector so that we can plot it later
        runtimes.append(runtime)
        scores.append(score)

        # Don't evaluate on the next size data set if this one took too long
        print("%0.003f , %0.3f s, %0.3f" % (percent, runtime, score))
        if timelimit!=-1: 
            if runtime > timelimit:
                break


    print("")
    return amt_of_data, runtimes, scores

#=================================================
#-------------------------------------------------
#=================================================

def algos_error_vs_data(clfs, x, y, x_test, y_test, clf_names=0, percent_of_data = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], timelimit=-1):

    amount_of_data = []
    times = []
    scores = []
    
    # If names is not supplied then use a list of numbers as the names
    if clf_names==0:
        clf_names = []
        for i in range(0,len(clfs)):
            clf_names.append(str(i))

    # Plot all of the graphs for the different algorithms
    for i in range(0,len(clfs)):
        clf_name = clf_names[i]
        amt_of_data, runtimes, score = algo_error_vs_data(clfs[i], x, y, x_test, y_test, clf_names[i], percent_of_data=percent_of_data, timelimit=timelimit)

        # Save the results to plot later
        amount_of_data.append(amt_of_data)
        times.append(runtimes)
        scores.append(score)

    return amount_of_data, times, scores

#=================================================
#-------------------------------------------------
#=================================================

def plot_timing_vs_data(amount_of_data, timing, clf_names=0):

    # If names is not supplied then use a list of numbers as the names
    if clf_names==0:
        clf_names = []
        for i in range(0,len(amount_of_data)):
            clf_names.append(str(i))

    plt.title("Timing")
    plt.xlabel('Amount of Training Data')
    plt.ylabel('Time in Seconds')
    for i in range(0,len(amount_of_data)):
        clf_name = clf_names[i]
        plt.plot(amount_of_data[i], timing[i], label=clf_name)

    plt.legend(loc="bottom right")

#=================================================
#-------------------------------------------------
#=================================================

def plot_error_vs_data(amount_of_data, error, clf_names=0):
    # If names is not supplied then use a list of numbers as the names
    if clf_names==0:
        clf_names = []
        for i in range(0,len(amount_of_data)):
            clf_names.append(str(i))

    # Plot all of the accuracy graphs for the different algorithms
    plt.sca(accuracy_axis)
    plt.title("Accuracy")
    plt.xlabel('Amount of Training Data')
    plt.ylabel('Accuracy Score')
    for i in range(0,len(amount_of_data)):
        clf_name = clf_names[i]
        plt.plot(amount_of_data[i], error[i], label=clf_name)

    plt.legend(loc="bottom right")

#=================================================
#-------------------------------------------------
#=================================================

def plot_timing_and_error(amount_of_data, timing, error, clf_names=0):

    fig = plt.figure(figsize=(12,12))
    timing_axis = fig.add_subplot(2,1,1)
    accuracy_axis = fig.add_subplot(2,1,2)
    
    # If names is not supplied then use a list of numbers as the names
    if clf_names==0:
        clf_names = []
        for i in range(0,len(amount_of_data)):
            clf_names.append(str(i))

    # Plot all of the timing graphs for the different algorithms
    plt.sca(timing_axis)
    plt.title("Timing")
    plt.xlabel('Amount of Training Data')
    plt.ylabel('Time in Seconds')
    for i in range(0,len(amount_of_data)):
        clf_name = clf_names[i]
        plt.plot(amount_of_data[i], timing[i], label=clf_name)
    plt.legend(loc="bottom right")

    # Plot all of the accuracy graphs for the different algorithms
    plt.sca(accuracy_axis)
    plt.title("Accuracy")
    plt.xlabel('Amount of Training Data')
    plt.ylabel('Accuracy Score')
    for i in range(0,len(amount_of_data)):
        clf_name = clf_names[i]
        plt.plot(amount_of_data[i], error[i], label=clf_name)
    plt.legend(loc="bottom right")
    plt.show()

#============================================
# Timing-------------------------------------
#============================================

#============================================
# load data
#============================================

# Get the data set for algo timing info (this one has more events but unequal #'s of y values in each class) 
df_equal = pd.read_csv('../tree_cover_pjt/data/cover_multi.data', index_col=0)
#df_equal = pd.read_csv('../tree_cover_pjt/data/train_multi.data', index_col=0)
df_equal = df_equal.reindex(np.random.permutation(df_equal.index))

#==============================================
# organize data into testing and training sets
#==============================================

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
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2, random_state=0)

# The various classifiers we want to look at
bdt = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.3, max_leaf_nodes=20, max_features=None) 
rf = RandomForestClassifier(n_estimators=5000, max_leaf_nodes=100, n_jobs=-1)
svc_lin = SVC(C=1, cache_size=1000)
svc_rbf = SVC(C=100, gamma=0.001,  kernel='rbf', cache_size=1000)
lr = LogisticRegression()


clfs = [rf, svc_lin, svc_rbf, lr, bdt]
clf_names = ["Random Forest 5000 Trees", "Linear SVC", "RBF_SVC", "Logistic_Regression", "BDT 1000 Trees 20 Nodes"]

# Timing and error info
amount_of_data, times, scores = algos_error_vs_data(clfs, x_train, y_train, x_test, y_test, percent_of_data=[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5], clf_names=clf_names, timelimit=1000)

#amount_of_data = [[464, 2324, 4648, 23240], [464, 2324, 4648, 23240]]
#times = [[0.7079291343688965, 1.8034718036651611, 2.0266530513763428, 9.019717931747437], [0.013071060180664062, 0.23213791847229004, 0.8129827976226807, 22.877432107925415]]
#scores = [[0.67955216302505095, 0.68850201802018884, 0.69826080221681019, 0.70143627961412358], [0.66935449170847572, 0.71741693415832641, 0.73975714912695889, 0.76392175761383097]]


print amount_of_data
print times
print scores


plot_timing_and_error(amount_of_data, times, scores, clf_names=clf_names)

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
