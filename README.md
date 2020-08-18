# Entropy for time series

Code to compute the entropy of a time series based on the maximum entropy principle as proposed by Prado et al. (2019) in https://aip.scitation.org/doi/pdf/10.1063/1.5125921 .
We found this entropy quantifier useful when dealing with ~10^4 (or more) time series as it has been shown to give qualitatively good results even without state space reconstruction (see F. Takens, Detecting strange attractors in turbulence (Springer, Berlin, 1981), pp. 366–381). 

In test.ipynb we provide an example using the Logistic map. Our computation uses a simple heuristic to speed up the algorithm and based on the observation that the entropy dependence on the parameter epsilon (i.e., threshold distance of the recurrence plot) does not change significantly for sample sizes of 1000 or 10000 microstates. We leverage this observation by testing the entropy dependence on epsilon using a sample of 1000 microstates. Once the optimal epsilon is chosen, we compute the final entropy using a sample size of 10000 microstastes for which the entropy converges.

In parallel.py we give a parallelized version of the code (using https://joblib.readthedocs.io/en/latest/). This is to be preferred when dealing with spatiotemporal datasets.

An application of this kind of entropy quantifiers can be found here:

Falasca, F., Crétat, J., Braconnot, P. and Bracco, A. Spatiotemporal complexity and time-dependent networks in sea surface temperature from mid- to late Holocene. Eur. Phys. J. Plus 135, 392 (2020). https://doi.org/10.1140/epjp/s13360-020-00403-x

Contacts: Fabrizio Falasca (fabrifalasca@gmail.com)
