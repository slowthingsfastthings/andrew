##############################################
# histograms.py                              #
##############################################
# functions for plotting histograms          #
#                                            #
#                                            #
##############################################

#============================================
# import
#============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import NullFormatter, MaxNLocator
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import radviz
from mi import mutual_info_matrix



#============================================
# various histogram plotting functions
#============================================

def histogram(col, axis=0, bins=25, maxvalues=41, color='blue', alpha=0.8, orientation='vertical'):
# Plot a 1d histogram from pandas

    num_unique_values = col.value_counts().size
    colmin = col.min()
    colmax = col.max()

    # If there are many unique values for the var, group them into bins
    if num_unique_values > maxvalues:
        bins=bins

    # If there are few unique values each bin may represent a value
    else:
        bins = num_unique_values
        colmax+=1

    # Change the active plotting axis
    if axis!=0:
        plt.sca(axis)

    plt.hist(col, color=color, alpha=alpha, bins=bins, range=(colmin, colmax), orientation=orientation)   

#------------------------------------------------
# ===============================================
#------------------------------------------------

def histogram_all_vars(df):
    # Keep track of the subplot we are in
    i=1
    
    # Make a figure and declare the size
    fig = plt.figure(figsize=(15, 12))

    # Make histograms for each column and put them in the figure
    for col in df.columns:
        current_axis = fig.add_subplot(5, 3, i)
        current_axis.set_title(col)
        histogram(df[col], axis=current_axis)
        i+=1
    fig.tight_layout()
    return fig

#------------------------------------------------
# ===============================================
#------------------------------------------------

def histogram_2d(x, y, axis=0, bins=(25,25), maxvalues=42):
# Make a 2d histogram from the columns, x,y, of a data frame

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    # The number of unique values in each vector
    xcounts = x.value_counts().size
    ycounts = y.value_counts().size

    # If there are not many unique values we can have a bin per value
    if xcounts < maxvalues:
        bins = (bins[0], xcounts)
        xmax = xmax+1
    if ycounts < maxvalues:
        bins = (ycounts, bins[1]) 
        ymax = ymax+1

    # Use numpy to make the 2D histogram then plot the histogram appropriately
    hist,xedges,yedges = np.histogram2d(y,x,bins=bins)
    color_bar=0

    # plot the histogram on a certain axis
    if axis!=0:
        color_bar = axis.imshow(np.log(hist), extent=[xmin,xmax,ymin,ymax], interpolation='nearest', aspect='auto', origin='lower')
        axis.grid(1)
        axis.set_xlabel(x.name)
        axis.set_ylabel(y.name)

    # plot the histogram through the main plotting system
    else:
        color_bar = plt.imshow(np.log(hist), extent=[xmin,xmax,ymin,ymax], interpolation='nearest', aspect='auto', origin='lower')
        plt.grid(1)
        plt.xlabel(x.name)
        plt.ylabel(y.name)

    # Set axis limits
    #axis.set_xlim(xmin,xmax)
    #axis.set_ylim(ymin,ymax)
    #plt.show()

#------------------------------------------------
# ===============================================
#------------------------------------------------

def histogram_2d_all_vars(df, target, bins=(25,25), maxvalues=42): 
    # Keep track of the subplot we are in
    i=1
    
    # Make a figure and declare the size
    fig = plt.figure(figsize=(15, 12))

    # Make histograms for each column and put them in the figure
    for col in df.columns:
        current_axis = fig.add_subplot(5, 3, i)
        current_axis.set_title(col)
        histogram_2d(df[col], target, axis=current_axis, bins=bins, maxvalues=maxvalues)
        i+=1
    fig.tight_layout()
    return fig
    
#------------------------------------------------
# ===============================================
#------------------------------------------------

def histogram_2d_with_projections(x, y, bins=(25,25), maxvalues=42):
    # Define the locations for the axes
    left, width = 0.07, 0.55
    bottom, height = 0.07, 0.55
    bottom_h = left_h = left+width+0.05

    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.30] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.30, height] # dimensions of y-histogram

    # Set up the axes for the plots
    fig = plt.figure(1, figsize=(12.5,10))
    axTemperature = fig.add_axes(rect_temperature)
    axHistx = fig.add_axes(rect_histx)
    axHisty = fig.add_axes(rect_histy)

    # Set gridlines for all the plots
    axTemperature.grid(1)
    axHistx.grid(1)
    axHisty.grid(1)

    # Plot the 2D histogram on its axes
    histogram_2d(x,y,axTemperature,bins=bins, maxvalues=maxvalues)

    binsx = bins[1]
    binsy = bins[0]

    # Plot the 1D histograms on their axes
    histogram(x, axis=axHistx, bins=binsx, color='blue', alpha=1)
    histogram(y, axis=axHisty, bins=binsy, color='red', alpha=1, orientation='horizontal')

    #Cool trick that changes the number of tickmarks for the histogram axes
    axHisty.xaxis.set_major_locator(MaxNLocator(4))
    #axHistx.yaxis.set_major_locator(MaxNLocator(4))

    # Large axis titles
    axTemperature.set_xlabel(x.name,fontsize=25)
    axTemperature.set_ylabel(y.name,fontsize=25)

    # Align the xaxes of the temp and xhist plots
    axTemperature.set_xlim(x.min(),x.max())
    axHistx.set_xlim(x.min(),x.max())

#------------------------------------------------
# ===============================================
#------------------------------------------------

def p_y_given_x(x, y, axis=0, bins=(25,25), maxvalues=42):
# Make a 2d histogram from the columns, x,y, of a data frame

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    # The number of unique values in each vector
    xcounts = x.value_counts().size
    ycounts = y.value_counts().size

    # If there are not many unique values we can have a bin per value
    if xcounts < maxvalues:
        bins = (bins[0], xcounts)
        xmax = xmax+1
    if ycounts < maxvalues:
        bins = (ycounts, bins[1]) 
        ymax = ymax+1

    # Use numpy to make the 2D histogram then plot the histogram appropriately
    hist,xedges,yedges = np.histogram2d(y,x,bins=bins)

    # Here we form p_y_given_x by dividing n_x_y by n_x
    normalized = pd.DataFrame(hist)
    normalized = normalized/normalized.sum()
    color_bar=0

    # plot the histogram on a certain axis
    if axis!=0:
        color_bar = axis.imshow(normalized, extent=[xmin,xmax,ymin,ymax], interpolation='nearest', aspect='auto', origin='lower')
        axis.grid(1)
        axis.set_xlabel(x.name)
        axis.set_ylabel(y.name)

    # plot the histogram through the main plotting system
    else:
        color_bar = plt.imshow(normalized, extent=[xmin,xmax,ymin,ymax], interpolation='nearest', aspect='auto', origin='lower')
        plt.grid(1)
        plt.xlabel(x.name)
        plt.ylabel(y.name)

#------------------------------------------------
# ===============================================
#------------------------------------------------

def p_y_given_x_all_vars(df, target, bins=(25,25), maxvalues=42): 
    # Keep track of the subplot we are in
    i=1
    
    # Make a figure and declare the size
    fig = plt.figure(figsize=(15, 12))

    # Make histograms for each column and put them in the figure
    for col in df.columns:
        current_axis = fig.add_subplot(5, 3, i)
        current_axis.set_title(col)
        p_y_given_x(df[col], target, axis=current_axis, bins=bins, maxvalues=maxvalues)
        i+=1
    fig.tight_layout()
    return fig


#------------------------------------------------
# ===============================================
#------------------------------------------------

def mutual_information(df, target_name, axis=0, bins=10):
    mi = pd.DataFrame(mutual_info_matrix(df.values,bins))
    mi.columns = df.columns
    mi.index = df.columns

    # Change the active plotting axis
    if axis!=0:
        plt.sca(axis)

    mi[target_name].plot(kind='barh', title='Mutual Information')    

    return mi

#------------------------------------------------
# ===============================================
#------------------------------------------------

def multidimensional_plots(df, target_name, maxevents=10000):
    # normalize
    df_std = (df - df.mean())/df.std()
    # put the unnormalized target back
    df_std[target_name] = df[target_name]
    # randomize the data frame order
    df_random = df_std.reindex(np.random.permutation(df_std.index))

    # make sure this doesn't take too long
    if df_random.shape[0] > maxevents:
        df_random = df_random[:maxevents]
    
    # Make a figure and declare the size
    fig = plt.figure(figsize=(9, 9))

    # Make histograms for each column and put them in the figure
    current_axis = fig.add_subplot(2, 2, 1)
    current_axis.set_title('Andrews Curves')
    andrews_curves(df_random, target_name, ax=current_axis)

    current_axis = fig.add_subplot(2, 2, 2)
    current_axis.set_title('Parallel Coordinates')
    parallel_coordinates(df_random, target_name, ax=current_axis, colormap='gist_rainbow')

    current_axis = fig.add_subplot(2, 2, 3)
    current_axis.set_title('Radviz Spring Tension')
    radviz(df_random, target_name, ax=current_axis, colormap='jet')

    #fig.tight_layout()
    return fig


df = pd.read_csv('../tree_cover_pjt/data/train_multi.data', index_col=0)
