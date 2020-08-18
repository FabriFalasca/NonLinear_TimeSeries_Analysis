# Entropy for time series

Code to compute the entropy of a time series based on the maximum entropy principle as proposed by Prado et al. (2019) in https://aip.scitation.org/doi/pdf/10.1063/1.5125921 
We found this entropy quantifier useful when dealing with ~10^4 (or more) time series as it has been shown to give qualitatively good results even without state space reconstruction (see F. Takens, Detecting strange attractors in turbulence (Springer, Berlin, 1981), pp. 366–381). 

In test.ipynb we show how to use the code in the context of Logistic map.

In parallel.py we give a parallelized version of the code (using https://joblib.readthedocs.io/en/latest/). This is to be preferred if dealing with spatiotemporal datasets.

An application of this kind of entropy quantifiers can be found here:

Falasca, F., Crétat, J., Braconnot, P. et al. Spatiotemporal complexity and time-dependent networks in sea surface temperature from mid- to late Holocene. Eur. Phys. J. Plus 135, 392 (2020). https://doi.org/10.1140/epjp/s13360-020-00403-x

Contacts: Fabrizio Falasca (fabrifalasca@gmail.com)

