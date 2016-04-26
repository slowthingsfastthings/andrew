##############################################
# plot.py                                    #
##############################################
# make some plots to get to know the data    #
#                                            #
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

df = pd.read_csv('data/train.csv', index_col=0)
#mi.mutual_information(df.values, "TARGET")
mi = histos.mutual_information(df, df["TARGET"], bins=[25,25])
#mi[:20].plot(kind='barh', title='Mutual Information')

# get a dataframe of the top 15 mutual info variables 
dftopv = pd.DataFrame()
dflastv = pd.DataFrame()
dftopv = df[list(mi[:15].index)]
dflastv = df[list(mi[-15:].index)]

#histos.histogram_all_vars(dftopv)
#histos.p_y_given_x_all_vars(dftopv, dftopv['TARGET'])
#histos.p_x_given_y_all_vars(dftopv, dftopv['TARGET'])
histos.p_y_given_x_all_vars(dflastv, dftopv['TARGET'])
#histos.p_x_given_y_all_vars(dflastv, dftopv['TARGET'])

plt.show()

#df.ix[:5,0:10] # row begin:row end, column begin:column end
#num_unique_values = col.value_counts().size
#df = df.reindex(np.random.permutation(df.index))

