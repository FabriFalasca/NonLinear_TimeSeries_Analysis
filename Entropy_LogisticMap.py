#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:10:06 2019

@author: fabrizio
"""

# Recurrence Entropy
# Entropy from recurrence matrix microstates

### Import Libraries ###

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import functions_RecPlots
import itertools
from sklearn.feature_extraction import image
import random
##########

# Author: Fab Falasca
# email: fabrifalasca@gmail.com

# Test using the Logistic Map
# Function (Logistic Map function)
#
def logMap(r,x0,n):
    '''
    Returns a time series of length n using the logistic map
        x_(n+1) = r*x_n(1-x_n) at parameter r and using the initial condition x0
    '''
    y = np.zeros(n+1)
    y[0] = x0
    for i in range(n):
        y[i+1] = r * y[i] * (1 - y[i])

    return y

epsilon = 0.13
entropy = np.array([], float)

for i in np.arange(3.50,4.0001,0.001):

    #  Parameters of logistic map
    r = i  # Bifurcation parameter
    x0 = 0.6   # Initial value
    #  Length of the time series
    length = 2000

    #  Create a time series using the logistic map
    time_seriesWithT = logMap(r,x0,length) # Initial transient still has to be removed
    time_series = time_seriesWithT[1001:] # Transient removed (transient considered as the first 1000 points)

    #  Settings for the recurrence plot
    # Dimension of a microstate
    n = 4
    # How many samples?
    sampleSize = 1000

    computedEntropy = functions_RecPlots.recurrenceEntropyFAST(time_series, epsilon,n,sampleSize)
    entropy = np.append(entropy, computedEntropy)

# To save
#np.savetxt('/Users/fabrizio/entropy.txt', entropy, fmt = '%1.6f')

# want to rapidly plot something?
def plotFig(x):
    fig = plt.figure(figsize=(8,6))
    #ax = fig.add_subplot(111)
    plt.xlabel('r', fontsize=18)
    plt.ylabel('Entropy', fontsize=16)
    plt.plot(x)
    plt.show()

# want to rapidly plot something?
def plotFig2D(x):
    fig = plt.figure(figsize=(8,6))
    plt.imshow(x,interpolation = 'none', cmap=plt.get_cmap('binary'))
    plt.colorbar()
    plt.show()
