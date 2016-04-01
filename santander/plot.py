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
mi[:20].plot(kind='barh', title='Mutual Information')
#df.ix[:5,0:10] # row begin:row end, column begin:column end
#num_unique_values = col.value_counts().size
#df = df.reindex(np.random.permutation(df.index))

