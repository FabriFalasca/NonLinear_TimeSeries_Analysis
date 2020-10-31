# Functions for computations of
    # (1) Function for computing the entropy of a time series
    # (4) Maximum entropy principle S_max: this is the entropy
    #     quantifier we want to use
    # (5) Compute S_max for all grid points in a spatiotemporal grid

# Fabri Falasca
# fabrifalasca@gmail.com

# functions to create recurrence plot
import numpy as np
#import matplotlib.pyplot as plt
from scipy import signal, spatial
from scipy.spatial.distance import pdist,cdist, squareform
from sklearn.feature_extraction import image
import itertools
import random

# Function to compute the recurrence entropy in a faster way
# Instead of computing the RP, I compute directly the microstates.
# (Better to use when for long timeseries)
def recurrenceEntropy(timeSeries, eps,n,sampleSize):
    # dimension of a microstate: n^2
    # Consider all Patches of dimension n^2
    # length of the time series
    dimT = len(timeSeries)
    # Initialize microstates
    micro = np.array([], float)

    # If the values are nan give a nan back
    if np.isnan(timeSeries[0]):

        normalizedEntropy = np.nan
    
    # If the values are not nan BUT the time series have std = 0 give 0
    elif ~np.isnan(timeSeries[0]) and np.std(timeSeries) == 0:

        normalizedEntropy = 0

    else:

        for i in range(sampleSize):
            # Bounds
            a = random.randint(0, dimT-n)
            b = random.randint(0, dimT-n)
            # Consider 4 consecutive times  t(n),t(n+1),t(n+2),t(n+3)
            # Choose the randomly. Then do the same t(n),t(n+1),t(n+2),t(n+3)
            # All distances between all points in the time series
            d = cdist(timeSeries[a:a+n,None],timeSeries[b:b+n,None],metric='euclidean')
            # Heaviside function
            d[d<=eps]=1
            #d[(d>eps)&(d!=1)]=0
            d[d!=1]=0
            micro = np.append(micro, d)
            # reshape the microstates
        microstates = micro.reshape((sampleSize,n,n))
        occurenceList = np.unique(microstates,axis=0, return_counts=1)[1]
        # From a list of occurences I compute the probability of
        probabilityList = occurenceList/float(sampleSize)
        # given the probability list we can define a recurrence entropy based on microstates in RP
        # Step (1) remove all zeros
        noZerosProbabilityList =  probabilityList[probabilityList>0]
        # Step (2) compute the Shannon Entropy of the Probability of occurence of those microstates
        entropy = - np.sum(noZerosProbabilityList * np.log(noZerosProbabilityList))
        # Max Entropy (found analytically)
        maxEntropy = np.log(sampleSize)
        # Normalized entropy
        normalizedEntropy = entropy/maxEntropy

    return normalizedEntropy

# Computing the maximum entropy principle
# Inputs: - timeSeries: timeseries
#         - n: size of the microstate
#         - sampleSize_h: number of samples to use for the heuristic
#           after many tests, sampleSize_h=1000 works best
#         - sampleSize for the final entropy computation
#           all tests showed convergence at sampleSize = 10000
def s_max(timeSeries, n, sampleSize_h, sampleSize):

    # Entropy for the heuristic
    entropy_ts_h = np.array([],float)
    # epsilons to test
    # We compute epsilons from 0 to 150% of the standard deviation of the time series
    epsilons = np.arange(0,1.55,0.05) * np.std(timeSeries)

    for i in np.arange(len(epsilons)):
        entropy_ts_h = np.append(entropy_ts_h,recurrenceEntropy(timeSeries, epsilons[i],n,sampleSize_h))

    # Find the epsilon that maximize the entropy
    #eps = epsilons[entropy_ts_h == np.max(entropy_ts_h)][0]
    eps = epsilons[entropy_ts_h == np.max(entropy_ts_h)]

    # Use this epsilon to compute the final entropy
    final_entropy = recurrenceEntropy(timeSeries, eps,n,sampleSize)

    return final_entropy


# Function to compute the s_max over an entire field
# Computing the maximum entropy principle
# Inputs: - timeSeries: spatiotemporal field. Format: [time, lats, lons]
#         - n: size of the microstate
#         - sampleSize_h: number of samples to use for the heuristic
#           after many tests, sampleSize_h=1000 works best
#         - sampleSize for the final entropy computation
#           all tests showed convergence at sampleSize = 10000
def recurrenceEntropyField(field,n,sampleSize_h, sampleSize):

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
                entropyField[i,j] = s_max(field[:,i,j], n, sampleSize_h, sampleSize)


    return entropyField
