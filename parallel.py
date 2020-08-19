# Author
# Fabri Falasca
# fabrifalasca@gmail.com

import numpy as np
import numpy.ma as ma
import netCDF4
from netCDF4 import Dataset
from utils_entropy import *

from joblib import Parallel, delayed
import multiprocessing

# *************************** Parameters **********************************

# If you want to use all cores 
#num_cores = multiprocessing.cpu_count()

# In case you want to specify the number of cores
num_cores = 60
# Dimension of a microstate
n = 4
# Sample size used for the Heuristics
sampleSize_h = 1000
# Sample size used for the final entropy computation
sampleSize = 10000

# path to nc file
path_nc_file = '../sst_a.nc'

# *************************************************************************

# We need to import the nc files to define the longitudes and latitudes

# Function to import netcdf datasets

def importNetcdf(path,variable_name):
    nc_fid = Dataset(path, 'r')
    field = nc_fid.variables[variable_name][:]
    return field

lon = importNetcdf(path_nc_file,'lon')
lat = importNetcdf(path_nc_file,'lat')
sst = importNetcdf(path_nc_file, 'tos')

dimT = np.shape(sst)[0]
dim_lat = len(lat)
dim_lon = len(lon)

# We want to be sure that if a time series has even just 1 nan inside, it should be always masked
for i in range(dim_lat):
    for j in range(dim_lon):
        if np.isnan(np.sum(sst[:,i,j])):
            sst[:,i,j] = np.nan

# Here I prefer not to work with masked array but simply with numpy array
sst = np.ma.filled(sst.astype(float), np.nan)

# To use the code in parallel we need to flatten the data
flat_data = sst.reshape(dimT,dim_lat*dim_lon).transpose()

import time
start_time = time.time()

entropy_flattened = Parallel(n_jobs=num_cores)(delayed(s_max)(i,n,sampleSize_h,sampleSize) for i in flat_data)

print("--- %s seconds ---" % (time.time() - start_time))

entropy_flattened = np.array(entropy_flattened)

entropy = entropy_flattened.reshape(dim_lat,dim_lon)

np.savetxt('./entropy.txt',entropy,fmt = '%1.6f')
