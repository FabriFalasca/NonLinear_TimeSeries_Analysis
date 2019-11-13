# Useful functions are coded up here
### Import Libraries ###

import numpy as np
#import matplotlib.pyplot as plt
from scipy import signal
# The libraries below are borrowed from here http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
##########

# Author: Fab Falasca
# email: fabrifalasca@gmail.com

##########

# Here I define a function to find the position of a number
# in a list given a certain constraint
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

##########

# Function to import a netcdf file in Python

def importNetcdf(path,variable_name):
    
    # The variable name should be written as this: 'variable'
    # Example: if the variable is SST, write 'SST'
    # Check on ncdump what are the names of the variables
    # Alternatively go here http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html
    # and define a ncdump function in Python
    # The path should be writtens as 'path'
    # Example /Fabri/test/file.nc: '/Fabri/test/file.nc'
    
    # Importing the netcdf file    
    nc_fid = Dataset(path, 'r')  # Dataset is the class behavior to open the file
                                 # and create an instance of the ncCDF4 class
    field = nc_fid.variables[variable_name] 
    
    return field                             


##########

# Function to compute anomalies: it removes seasonality and a linear trend
# It assumes: 
            # (1) Monthly time steps
            # (2) The total number of time steps has to be a multiple of 12 
            #     So, it assumes not missing values: if I have to remove the seasonality of 100 years
            #     I expect to have 1200 months as input
def detrendedAnomalies(x): 
    
    # STEP (1): REMOVE THE SEASONAL CYCLE
          
    # monthlyAverage will hold the averages of every month
    monthlyAverage = np.zeros(12)
    # Number of months
    dimT = len(x)
    # Number of years
    nYears = dimT/12.
    # Initialize the new time series
    ts = np.zeros(dimT)
    
    for i in range(0,dimT):
        k = i + 1 # since the counter in Python start from 0...fucking bastards!
        month = k % 12 # % is the Modulo operator 
        if month == 0:
            monthlyAverage[11] = monthlyAverage[11] + x[i] # remember: should be 12, but Python start
                                                           #  counting from zero...
        else:
            monthlyAverage[month-1] = monthlyAverage[month-1] + x[i] # month -1 because the counter starts from 0
    
    monthlyAverage = monthlyAverage / nYears
    
    for i in range(0,dimT):        
        k = i + 1 # fixing the fact that the counter start from 0
        month = k % 12 # % is the Modulo operator
        if month == 0:
            ts[i] = x[i] - monthlyAverage[11]
        else:
            ts[i] = x[i] - monthlyAverage[month-1] # month -1 because the counter starts from 0
                                                                                    
    # STEP (2): REMOVE A LINEAR TREND
    detrendedTS = signal.detrend(ts)
    
    return detrendedTS

##########

# Function to remove the trend
            # (1) Yearly time step
def detrend(x): 
    
    detrendedTS = signal.detrend(x)
    
    return detrendedTS
    
##########
# Remove outliers from 1 time series
# I assume that every time series has a pdf that is (at first order) Gaussian
# I assign nan values to all points greater/smaller than the mean +/- 3 Standard Deviations

def removeOutlier(timeSeries):
    
    dimT = len(timeSeries)
    
    # Compute the mean
    mean = np.mean(timeSeries)
    # Compute the standard deviation
    std = np.std(timeSeries, ddof = 1)
    
    # Outliers are all points that are larger than mean + 3 std
    # or less than mean - 3 std
    # This makes lot of sense if the pdf is a normal distribution
    meanPlus3STD = mean + 3*std
    meanMinus3STD = mean - 3*std
    
    # Now we look at every point and if it is an outlier we equal it to nan
    # If I do it like this it modifies also the original time series...
    #timeSeries[timeSeries < meanMinus3STD ] = np.nan
    #timeSeries[timeSeries > meanPlus3STD ] = np.nan
    
    # Initialize a new time series with no outliers
    noOutlierTimeSeries = np.zeros(dimT)
    
    for i in range(dimT):
        
        if timeSeries[i] < meanMinus3STD or timeSeries[i] > meanPlus3STD:
            noOutlierTimeSeries[i] = np.nan
        else:
            noOutlierTimeSeries[i] = timeSeries[i]    
    
    return noOutlierTimeSeries
    
########## 
# I remove outliers from a spatio temporal field  

def fieldRemoveOutliers(field):
    
    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]
    
    # Initialize the anomaly field
    noOutliersField = np.zeros([dimTime,dimLat,dimLon])
    
    for i in range(0,dimLat):
        for j in range(0,dimLon):
            # Check if is a masked value
            if np.isnan(field[0,i,j]):
                noOutliersField[:,i,j] = field[:,i,j] # if it is mask just return the mask
            else:
                # if not masked return a detrended anomaly             
                noOutliersField[:,i,j] = removeOutlier(field[:,i,j]) 
                
    return noOutliersField
    
    
##########

# Function that given a 3D spatio temporal field (2 spatial dimensions and 1 time dimension)
# Compute an anomaly field
# The input is formatted in this way formatted[field] = {time, latitude, longitude}
# It assumes that the masked points are masked with nan
        
def fieldDetrend(field):
    
    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]
    
    # Initialize the anomaly field
    anomalyField = np.zeros([dimTime,dimLat,dimLon])
    
    for i in range(0,dimLat):
        for j in range(0,dimLon):
            # Check if is a masked value
            if np.isnan(field[0,i,j]):
                anomalyField[:,i,j] = field[:,i,j] # if it is mask just return the mask
            else:
                # if not masked return a detrended anomaly             
                anomalyField[:,i,j] = detrend(field[:,i,j]) 
                
    return anomalyField
    
# Function to linearly "step detrend"   a Field 
def stepDetrendField(field):
    
    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]
    
    # Initialize the "new" field where storing the detrended time series
    stepDetrendedField = np.zeros([dimTime,dimLat,dimLon])
    
    # Step (1)
    # Compute the trend every 100 years and define a new
    # spatio temporal field
    for i in np.arange(0,6000,100):
        
        if i == 0:
            start = i
            end = 100
        else:
            start = i - 1
            end = 100 + i - 1
            
        stepDetrendedField[start:end] = fieldDetrend(field[start:end])
        
    return stepDetrendedField     
        
    
# Function to compute epsilon given the COMPLETE time series
# (1) We consider time windows of 100 years and detrend them, attach all of them   
# to construct a new time series
# (2) given this new time series we complete epsilon as a percentage 
# of its max - min
# (3) we offer a similar way of computing epsilon but based on
# percentages of its standard deviation instead of its diameter
# For each time series in each grid point I compute epsilon and 
# save the result in a 2-D grid

def epsilonComputing(field,percentage):
    
    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]
    
    # Initialize the epsilon field
    # When computed using the diameter
    epsilonField_Diameter = np.zeros([dimLat,dimLon])
    # when computed using the standard deviation
    epsilonField_std = np.zeros([dimLat,dimLon])
    
    #(2) For every grid point, compute the minimum and max of a time series and compute epsilon
    # do then the same using the standard deviation
    for i in range(0,dimLat):
        for j in range(0,dimLon):
            # Check if is a masked value
            if np.isnan(field[0,i,j]):
                epsilonField_Diameter[i,j] = field[0,i,j] # if it is mask just return the mask
            else:
                # Compute epsilon as x percentage of diameter of the time series
                diameter = np.abs(np.nanmax(field[:,i,j]) - np.nanmin(field[:,i,j]))
                epsilon_diameter = percentage * diameter
                epsilonField_Diameter[i,j] = epsilon_diameter
                # Compute epsilon as x percentage of diameter of the time series
                std = np.nanstd(field[:,i,j], ddof = 1)
                epsilon_std = percentage * std                
                epsilonField_std[i,j] = epsilon_std
                
    return [epsilonField_Diameter, epsilonField_std]            
    
              

##########

# Function that given a 3D spatio temporal field (2 spatial dimensions and 1 time dimension)
# Compute an anomaly field
# The input is formatted in this way formatted[field] = {time, latitude, longitude}
# It assumes that the masked points are masked with nan
        
def fieldAnomaly(field):
    
    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]
    
    # Initialize the anomaly field
    anomalyField = np.zeros([dimTime,dimLat,dimLon])
    
    for i in range(0,dimLat):
        for j in range(0,dimLon):
            # Check if is a masked value
            if np.isnan(field[0,i,j]):
                anomalyField[:,i,j] = field[:,i,j] # if it is mask just return the mask
            else:
                # if not masked return a detrended anomaly             
                anomalyField[:,i,j] = detrendedAnomalies(field[:,i,j]) 
                
    return anomalyField    

##########

# Function that given a 3D spatio temporal field (2 spatial dimensions and 1 time dimension)
# Find the time delay parameter
# The delay time is computed using the autocorrelation method: tau is when r decreases of a 1/e  
# Mainly used for testing

def timeDelayField_AutocorrelationMethod(field):

    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]

    # Initialize the grid holding the time delays
    timeDelay = np.zeros([dimLat,dimLon])
   

    for i in range(0,dimLat):
        for j in range(0,dimLon):
            # Check if is a masked value
            if np.isnan(field[0,i,j]):
                timeDelay[i,j] = field[0,i,j] # if it is mask just return the mask
            else:
                # STEP (1): find the delay
                # Compute autocorrelation and delayed mutual information.
                r = delay.acorr(field[:,i,j], maxtau=100)
                # t is when r decreases of 1/e
                tau = np.argmax(r < 1.0 / np.e)
                # STEP (2): assign a value of tau to this time series
                timeDelay[i,j] = int(tau) # I return it as integer
                
    return timeDelay
    
##########

# Function that given a 3D spatio temporal field (2 spatial dimensions and 1 time dimension)
# Compute the spatial embedding
# The spatial embedding m is computed with the false nearest neighbors method

def spatialEmbedding(field, delay):

    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]

    # Initialize the grid holding the time delays
    spatialEmb = np.zeros([dimLat,dimLon])
   

    for i in range(0,dimLat):
        for j in range(0,dimLon):
            # Check if is a masked value
            if np.isnan(field[0,i,j]):
                spatialEmb[i,j] = field[0,i,j] # if it is mask just return the mask
            else:
                # STEP (1): find the delay
                # Compute the false nearest neighbors
                dim = np.arange(1, 10 + 1)
                f1, f2, f3 = dimension.fnn(field[:,i,j], tau=int(delay[i,j]), dim=dim, window=10, metric='cityblock')
                # Consider the first test f1
                m = np.argmax(f1 == 0) + 1
                # The dimension m is when the ffn goes to zero
                # STEP (2): assign a value of tau to this time series
                spatialEmb[i,j] = m
                
    return spatialEmb
    

    
##########

# Function that given a 3D spatio temporal field (2 spatial dimensions and 1 time dimension)
# Compute the entropy field: at every point the permutation entropy is computed
# The input is formatted in this way formatted[field] = {time, latitude, longitude}
# It assumes that the masked points are masked with nan

# The delay time is computed using the autocorrelation method                                                         
    
def entropyField(field,permutation_order):
    
    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]
    
    # Initialize the entropy field
    entropyField = np.zeros([dimLat,dimLon])
    
    for i in range(0,dimLat):
        for j in range(0,dimLon):
            # Check if is a masked value
            if np.isnan(field[0,i,j]):
                entropyField[i,j] = field[0,i,j] # if it is mask just return the mask
            else:
                # STEP (1): find the delay
                # Compute autocorrelation and delayed mutual information.
                r = delay.acorr(field[:,i,j], maxtau=100)
                # The delay is the point of max decrease in autocorr
                tau = int(np.argmax(r < 1.0 / np.e))
                # STEP (2): compute the permutation entropy of order 3               
                # if not masked return the permutation entropy             
                entropyField[i,j] = ent.permutation_entropy(field[:,i,j],order=permutation_order, delay=tau, normalize=False)
    
    return entropyField
    
def entropyFieldFixedDelay(field,permutation_order,fixedDelay):
    
    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]
    
    # Initialize the entropy field
    entropyField = np.zeros([dimLat,dimLon])
    
    for i in range(0,dimLat):
        for j in range(0,dimLon):
            # Check if is a masked value
            if np.isnan(field[0,i,j]):
                entropyField[i,j] = field[0,i,j] # if it is mask just return the mask
            else:
                # STEP (1): find the delay
                # Compute autocorrelation and delayed mutual information.
                #r = delay.acorr(field[:,i,j], maxtau=100)
                # The delay is the point of max decrease in autocorr
                #tau = int(np.argmax(r < 1.0 / np.e))
                # STEP (2): compute the permutation entropy of order 3               
                # if not masked return the permutation entropy             
                entropyField[i,j] = ent.permutation_entropy(field[:,i,j],order=permutation_order, delay=fixedDelay, normalize=False)
    
    return entropyField
    
##########

# Function that given a 3D spatio temporal field (2 spatial dimensions and 1 time dimension)
# Compute the Shannon entropy field: at every point the Shannon entropy is computed
# The input is formatted in this way formatted[field] = {time, latitude, longitude}
# It assumes that the masked points are masked with nan

# The delay time is computed using the autocorrelation method                                                         
    
def shannonEntropyField(field):
    
    # Dimensions of the field?
    # Time dimension
    dimTime = np.shape(field)[0]
    # Number of points in latitude (y-axis)
    dimLat = np.shape(field)[1]
    # Number of points in longitude (x-axis)
    dimLon = np.shape(field)[2]
    
    # Initialize the entropy field
    entropyField = np.zeros([dimLat,dimLon])
    
    for i in range(0,dimLat):
        for j in range(0,dimLon):
            # Check if is a masked value
            if np.isnan(field[0,i,j]):
                entropyField[i,j] = field[0,i,j] # if it is mask just return the mask
            else:           
                entropyField[i,j] = ent.shannon_entropy(np.round(field[:,i,j],2)) # I am rounding to 0.01 the data
    
    return entropyField                              
              
    
    
    
    
    
    
    
    
    
    
