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


# Should make the size of the figure and the number of rows and columns in the figures inputs to the functions
# Should make the log plot for the histograms an option

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

def standardize(df,savecol=''):
    # normalize
    if standardize==True:
        df_std = (df - df.mean())/df.std()
        # put the unnormalized target back
        if savecol!='':
            df_std[savecol] = df[savecol]
        df = df_std

#------------------------------------------------
# ===============================================
#------------------------------------------------

def multidimensional_plots(df, target_name, maxevents=10000, standardize=False):

    # randomize the data frame order
    df_random = df.reindex(np.random.permutation(df.index))[:maxevents]
    
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

#------------------------------------------------
# ===============================================
#------------------------------------------------

def which_interval(x,y,x_edges,y_edges):
# we have lists of the edges of a histogram and we check to see which bin the (x,y) pair falls in.
# the bin location is given by the x index and the y index, which we return.

    xindex = -1
    yindex = -1

    for i in range(len(x_edges)):
        if x < x_edges[i]:
            xindex = i-1
            break

    for i in range(len(y_edges)):
        if y < y_edges[i]: 
            yindex = i-1
            break

    return xindex, yindex

#------------------------------------------------
# ===============================================
#------------------------------------------------

def bin_time(x,y,z, bins=(25,25), maxvalues=42):
# Store a list of the z values that fall into each x,y bin.
# With a 2x2 binning we have something like
# [[zlist(0,0),zlist(0,1)],
#  [zlist(1,0),zlist(1,1)]]
# where [0][0] = zlist(0,0) = a list of all the z values that fell in this bin 
# e.g. [z_1,z_24,z_15, etc] all fell in the interval [x_edge0,x_edge1),[y_edge0,y_edge1)
# I define the row coordinates to be y and the columns to be x since y is usually the vertical axis and x the horizontal
# we can use this to find the median, mean, or mode of the z value in a bin and plot a histogram

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    zmin = z.min()
    zmax = z.max()
    
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

    # bins pair is (ybins,xbins) for imshow. we keep this convention here.
    xbins = bins[1]
    ybins = bins[0]

    xwidth = (xmax - xmin)/xbins
    ywidth = (ymax - ymin)/ybins

    # list of edges
    xedges = np.arange(xmin, xmax+xwidth, xwidth)
    yedges = np.arange(ymin, ymax+ywidth, ywidth)
    
    # matrix of the z value lists
    # Check to make sure this has the correct dimensions
    #z_matrix = [[[0,0,0] for x in range(xbins)] for y in range(ybins)]
    z_matrix = [[[0,0,0] for x in range(3)] for y in range(2)]
    z_matrix[1][2] = [1,2,3,4,5]
    
    print(z_matrix)

    # for each x,y bin we make a list of the z values that fell in this bin
    # later we can reduce the list of z values to its average, median, or mode
    #for point i 
    #    xi,yi = which_interval(x[i], y[i], x_edges, yedges) 
    #    z_matrix[yi][xi].append(z[i])

    #print(xedges)
    #print("---")
    #print(yedges)
    #print("---")

    # mesh grid, not sure how to use this yet
    #xi, yi = np.meshgrid(xedges,yedges)
    #print(xi)
    #print("---")
    #print(yi)
    #print("---")

    

def histogram_2d_xyz(x, y, z, axis=0, bins=(25,25), maxvalues=42, ztype='median'):
# Make a 2d histogram from the columns, x,y, of a data frame

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    zmin = z.min()
    zmax = z.max()

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

    # Use numpy to make the 2D histogram. This will have the count in each bin but we want some version of the z value. 
    hist,xedges,yedges = np.histogram2d(y,x,bins=bins)

    # Let's get the z value
    

    color_bar=0

    print(xedges)
    print(yedges)

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


# Make a 2d histogram from the columns, x,y, of a data frame
#df = pd.read_csv('../tree_cover_pjt/data/cover_multi.data', index_col=0)
df = pd.read_csv('../tree_cover_pjt/data/train_multi.data', index_col=0)
df = df.reindex(np.random.permutation(df.index))
x = df['Soil_Type']
y = df['Wilderness_Area']
z = df['Cover_Type']

bin_time(x,y,z)

