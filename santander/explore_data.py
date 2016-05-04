##############################################
# explore_data.py                            #
##############################################
# make some plots, run PCA, etc              #
# get to know the data                       #
#                                            #
##############################################

#============================================
# import
#============================================

import sys
sys.path.insert(0, '../lib')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import histograms as histos
from sklearn import decomposition

df = pd.read_csv('data/train.csv', index_col=0)
#mi.mutual_information(df.values, "TARGET")
mi = histos.mutual_information(df, df["TARGET"], bins=[25,25])
#mi[:20].plot(kind='barh', title='Mutual Information')

# get a dataframe of the top 15 mutual info variables 
dftopv = pd.DataFrame()
dftopv = df[list(mi[:15].index)]

# get a dataframe of the bottom 15 mutual info variables 
dflastv = pd.DataFrame()
dflastv = df[list(mi[-15:].index)]

print "\n#### Top mutual information vars...\n"
print(mi[:50])
print "\n#### Bottom mutual information vars...\n"
print(mi[-50:])
print "\n"

# standardize the variables if we care to
#df_std = (df - df.mean()) / df.std()
pca = decomposition.PCA(n_components=20)
pca.fit(df.ix[:, df.columns != 'TARGET'])

# Check out the eigenvectors and eigenvalues for PCA
print "#### PCA Eigenvalues...\n"
print(pca.explained_variance_ratio_)  #Eigenvalues
print "\n#### PCA Eigenvectors...\n"
eigenvectors = pca.components_        #Eigenvectors are the rows
print(eigenvectors)                   
print(eigenvectors.shape)


# Plot the top/bottom MI variables if we want
#histos.histogram_all_vars(dftopv)
#histos.p_y_given_x_all_vars(dftopv, dftopv['TARGET'])
#histos.p_x_given_y_all_vars(dftopv, dftopv['TARGET'])
#histos.p_y_given_x_all_vars(dflastv, dftopv['TARGET'])
#histos.p_x_given_y_all_vars(dflastv, dftopv['TARGET'])

plt.show()

#df.ix[:5,0:10] # row begin:row end, column begin:column end
#num_unique_values = col.value_counts().size
#df = df.reindex(np.random.permutation(df.index))

