### Import Libraries ###

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pyentrp import entropy as ent
from nolitsa import data, delay, noise, dimension
import functions
import functions_RecPlots
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
# The libraries below are borrowed from here http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
##########

# Author: Fab Falasca
# email: fabrifalasca@gmail.com

########## Import the data ##########


# Path to netcdf files 
path_file = '/Users/fabrizio/Dropbox/PHD/Phd/Project/Paris/Project/data/data_SST/yearly_SST.nc'
path_to_mask = '/Users/fabrizio/Dropbox/PHD/Phd/Project/Paris/Project/data/data_SST/fract_oce_LOWRES_GLOBAL.nc'  # path to mask
# Importing SST and the mask
Temp_not_masked = functions.importNetcdf(path_file,'tsol_oce')
mask = functions.importNetcdf(path_to_mask,'fract_oce')[0]
# mask goes from 0 to 1. We want the mask to be binary
# 1 in the sea, everything else is -10000
mask[mask < 1] = -10000
# The temperature field has to be masked
Temp = mask * Temp_not_masked
# Everything below -1000 go to nan
Temp[Temp < -1000] = np.nan

# To compute epsilon we define a new field, step detrended and with no outliers
temperatureStepDetrended = functions.stepDetrendField(Temp)
temperatureStepDetrended_No_Outliers = functions.fieldRemoveOutliers(temperatureStepDetrended)

# Compute epsilon as 30% of the standard deviation of the total time series
percent = 0.3
computedEpsilonField = functions.epsilonComputing(temperatureStepDetrended_No_Outliers,percent)
computedEpsilonField_std = computedEpsilonField[1]

for i in np.arange(0,5900 + 10,10):
    
    if i == 0:
        start = i
        end = 100
    else:
        start = i - 1
        end = 100 + i - 1

    # I consider points from start to end, and detrend linearly the field
    ts = functions.fieldDetrend(Temp[start:end])
    # Compute the recurrence entropy
    recurrenceEntropyField = functions_RecPlots.recurrenceEntropyField(ts,computedEpsilonField_std,4,5000)

    # The period is named as i + 1
    period = i + 1
    np.savetxt('/Users/fabrizio/Dropbox/PHD/Phd/Project/Paris/Project/Yearly_Time_Step/SST/100Years_Windows/sst/Entropy_SST_100Years/recurrenceEntropy_'+str(period)+'.txt',recurrenceEntropyField, fmt = '%1.6f')

'''
# want to rapidly plot something?        
def plotFig(x):
    fig = plt.figure(figsize=(8,6))
    #ax = fig.add_subplot(111)
    plt.plot(x)
    plt.show()
    
# want to rapidly plot something?        
def plotFig2D(x):
    fig = plt.figure(figsize=(8,6))
    #ax = fig.add_subplot(111)
    plt.imshow(x)
    plt.colorbar()
    plt.show()    
'''



                                                               

