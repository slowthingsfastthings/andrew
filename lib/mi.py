# Method to calculate mutual information between variables
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

#bins : int or array_like or [int, int] or [array, array], optional
#The bin specification:
#  If int, the number of bins for the two dimensions (nx=ny=bins).
#  If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
#  If [int, int], the number of bins in each dimension (nx, ny = bins).
#  If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
#  A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.

def calc_MI(x, y, bins=[25,25], maxvalues=41):
    "Calculates the mutual information between two variables."
    # [0] gets the histogram, [1] & [2] are binning info.

    num_unique_values_x = pd.value_counts(x).size
    num_unique_values_y = pd.value_counts(y).size
    if num_unique_values_x < maxvalues: 
        bins[0] = num_unique_values_x
    if num_unique_values_y < maxvalues: 
        bins[1] = num_unique_values_y
	    
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    # use scikit learn to calculate the mi value
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def mutual_info_matrix(dfvalues, bins=[25,25], maxvalues=41):
    # calculate the mutual information between all of the different variables
    # matMI[1,2] = mutual info between variables 1 and 2
    # get the number of different variables
    n = dfvalues.shape[1]

    # initialize the matrix with the correct size
    matMI = np.zeros((n, n))
  
    # calculate the mutual info between each pair of variables
    # each column in matrix A gives a vector of values for some variable
    # i.e. column 1 will have the event 0's x1 value in the 0th location, event 1's x1 value in the 1st location, etc
    for ix in np.arange(n):
        for jx in np.arange(ix,n):
            matMI[ix,jx] = calc_MI(dfvalues[:,ix], dfvalues[:,jx], bins=bins, maxvalues=maxvalues)

    # make the matrix symmetric by adding the transpose. Subtract off the diagonals to prevent doubling these values.
    matMI+=(matMI.T-np.diagflat(matMI.diagonal()));
    return matMI

def mutual_info_matrix(dfvalues, targetvalues, bins=[25,25], maxvalues=41):
    # calculate the mutual information between all of the different variables
    # matMI[1,2] = mutual info between variables 1 and 2
    # get the number of different variables
    n = dfvalues.shape[1]

    # initialize the matrix with the correct size
    mi = np.zeros(n)
  
    # calculate the mutual info between each pair of variables
    # each column in matrix A gives a vector of values for some variable
    # i.e. column 1 will have the event 0's x1 value in the 0th location, event 1's x1 value in the 1st location, etc
    for ix in np.arange(n):
        mi[ix] = calc_MI(dfvalues[:,ix], targetvalues, bins=bins, maxvalues=maxvalues)

    mi = mi/mi[n-1]

    return mi
