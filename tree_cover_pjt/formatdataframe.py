import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def manyBoolToSingleMulticlass(df, regx):
    "Compress many boolean columns into a single multivalued column"
    #Get the columns to compress via regex
    reduced = df.filter(regex=regx)
    shape = reduced.shape
    rows = shape[0]
    cols = shape[1]
    #Scale the boolean 1 by the column number to create the multivalues
    reduced = reduced*range(1,cols+1)
    #Reduce to a single column by multiplying every row by the vector (1,1,1,...,1)
    #Since only one column can be true we get 1*0+1*0+1*X+1*0+...+1*0 = X
    ones = np.ones((cols,1))
    reduced = np.dot(reduced,ones)
    #We have a one column vector. Turn this into a single array so we can pass it to DataFrame.
    reduced = reduced.T[0]
    final = pd.Series(reduced)
    return final

# Read CSV into DataFrame, the first column is the index column
df = pd.read_csv("./data/train.csv", index_col=0);

# Remap the DF indicies to start at 0 instead of 1
df.index = range(0,df.shape[0])

# Reduce the many boolean columns for soil and wilderness area into single columns
reduced_soil = manyBoolToSingleMulticlass(df, 'Soil') #40 boolean types -> 1 type with 40 values
reduced_wilderness_area = manyBoolToSingleMulticlass(df, 'Wilderness_Area')#4 boolean types -> 1 type with 4 values

# Merge the new and improved columns with the other variables from the original.
dfbegin = df.iloc[:,0:10]
dfend = df['Cover_Type']
dfwildarea = pd.DataFrame({'Wilderness_Area': reduced_wilderness_area})
dfsoiltype = pd.DataFrame({'Soil_Type': reduced_soil})

# The final product
df_final = pd.concat([dfbegin,dfwildarea,dfsoiltype,dfend],axis=1)

# Make a dataframe with standardized variables.
#df_std = (df_final - df_final.mean())/df_final.std()

df_final.to_csv('data/train_multi.data')
#df_std.to_csv('data/cover_multi_std.data')
