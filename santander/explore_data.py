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
from mi import mutual_information
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data/train.csv', index_col=0)

#print "#### Dataframe Counts...\n"
#with pd.option_context('display.max_rows', 999, 'display.max_columns', 3):
#    print(df.apply(pd.Series.nunique))

# standardize the variables if we care to
#df_std = scale(df)
#pca = decomposition.PCA(n_components=100)
#x = df.ix[:, df.columns != 'TARGET']
#y = df['TARGET']
#pca.fit(x)
#
## Check out the eigenvectors and eigenvalues for PCA
#print "\n#### PCA Eigenvalues...\n"
#print(pca.explained_variance_ratio_)  #Eigenvalues
#print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)) # see how many pca vectors we need to explain all of the variance (looks like 13)
#print "\n#### PCA Eigenvectors...\n"
#eigenvectors = pca.components_        #Eigenvectors are the rows
#print(eigenvectors)                   
#print(eigenvectors.shape)
#
##print "\n#### x before PCA transformation...\n"
##print(x.ix[:,:3])
#dfpca = pd.DataFrame(pca.transform(x))
##print "\n#### x after PCA transformation...\n"
#dfpca['TARGET'] = pd.Series(y.values)
##print(dfpca.ix[:,:3])
##print(dfpca['TARGET'])
#
##df = dfpca

#mi.mutual_information(df.values, "TARGET")
mi = mutual_information(df, df['TARGET'], bins=[25,25])
#mi[:15].plot(kind='barh', title='Mutual Information')
#plt.show()

# get a dataframe of the top 15 mutual info variables 
dftopv = pd.DataFrame()
dftopv = df[list(mi[:50].index)]

# get a dataframe of the bottom 15 mutual info variables 
dflastv = pd.DataFrame()
dflastv = df[list(mi[-15:].index)]

x = dftopv.ix[:, dftopv.columns != 'TARGET']
scale(x)
y = dftopv['TARGET']
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2, random_state=0)
weights = {0:0.04, 1:0.96}
lr = LogisticRegression(C=1.0, class_weight=weights)

lr.fit(x_train,y_train)

y_true, y_scores = y_test, lr.predict_proba(x_test)
score = roc_auc_score(y_test, y_scores[:,1])

print "\n#### ROC AUC Score...\n"
print score

#print "\n#### Top mutual information vars...\n"
#print(mi[:15])
#print "\n#### Bottom mutual information vars...\n"
#print(mi[-15:])
#print "\n"

# Plot the top/bottom MI variables if we want
#histos.histogram_all_vars(dftopv)
#histos.p_y_given_x_all_vars(dftopv, dftopv['TARGET'])
#histos.p_x_given_y_all_vars(dftopv, dftopv['TARGET'])
#histos.p_y_given_x_all_vars(dflastv, dftopv['TARGET'])
#histos.p_x_given_y_all_vars(dflastv, dftopv['TARGET'])

#plt.show()

#df.ix[:5,0:10] # row begin:row end, column begin:column end
#num_unique_values = col.value_counts().size
#df = df.reindex(np.random.permutation(df.index))

