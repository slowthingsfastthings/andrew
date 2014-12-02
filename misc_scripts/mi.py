# Method to calculate mutual information between variables
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bins):
    "Calculates the mutual information between two variables."
    # [0] gets the histogram, [1] & [2] are binning info.
    c_xy = np.histogram2d(x, y, bins)[0]
    # use scikit learn to calculate the binning information
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def mutual_info_matrix(A, bins):
    # get the number of different variables
    n = A.shape[1]

    # initialize the matrix with the correct size
    matMI = np.zeros((n, n))
  
    # calculate the mutual info between each pair of variables
    for ix in np.arange(n):
        for jx in np.arange(ix,n):
            matMI[ix,jx] = calc_MI(A[:,ix], A[:,jx], bins)

    # make the matrix symmetric by adding the transpose. prevent doubling up the diagonals.
    matMI+=(matMI.T-np.diagflat(matMI.diagonal()));
    return matMI
