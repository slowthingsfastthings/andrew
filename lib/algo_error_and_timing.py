##############################################
# algo_error_and_timing.py                   #
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

# Sci kit learn evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Timing
import time

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

def algo_error_vs_data(clf, x, y, x_test, y_test, clf_name="", percent_of_data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], score_type='accuracy', timelimit=-1):
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
        #print(y_test)
        #print(y_scores)
        #print(y_scores[:,1])
        if score_type == 'roc':
            y_scores = clf.predict_proba(x_test)
            score = roc_auc_score(y_test, y_scores[:,1])
        if score_type == 'accuracy':
            y_pred = clf.predict_proba(x_test)
            score = accuracy_score(y_test, y_pred)
        if score_type == 'f1':
            y_pred = clf.predict_proba(x_test)
            score = f1_score(y_test, y_pred, average='weighted')

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

def algos_error_vs_data(clfs, x, y, x_test, y_test, clf_names=0, percent_of_data = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], score_type='accuracy', timelimit=-1):

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
        amt_of_data, runtimes, score = algo_error_vs_data(clfs[i], x, y, x_test, y_test, clf_names[i], percent_of_data=percent_of_data, score_type=score_type, timelimit=timelimit)

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

def plot_error_vs_data(amount_of_data, error, score_type='accuracy', clf_names=0):
    # If names is not supplied then use a list of numbers as the names
    if clf_names==0:
        clf_names = []
        for i in range(0,len(amount_of_data)):
            clf_names.append(str(i))

    # Plot all of the accuracy graphs for the different algorithms
    plt.sca(accuracy_axis)
    plt.title(score_type)
    plt.xlabel('Amount of Training Data')
    plt.ylabel(score_type)
    for i in range(0,len(amount_of_data)):
        clf_name = clf_names[i]
        plt.plot(amount_of_data[i], error[i], label=clf_name)

    plt.legend(loc="bottom right")

#=================================================
#-------------------------------------------------
#=================================================

def plot_timing_and_error(amount_of_data, timing, error, score_type='accuracy', clf_names=0):

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
    plt.title(score_type)
    plt.xlabel('Amount of Training Data')
    plt.ylabel(score_type)
    for i in range(0,len(amount_of_data)):
        clf_name = clf_names[i]
        plt.plot(amount_of_data[i], error[i], label=clf_name)
    plt.legend(loc="bottom right")
    plt.show()
