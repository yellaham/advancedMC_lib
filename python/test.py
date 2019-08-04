import numpy as np
import scipy.stats as sp
import ais_lib

# Define target distribution
dim = 2
gauss_target = lambda x: sp.multivariate_normal.logpdf(x, mean=np.zeros(dim), cov=np.eye(dim))

# Run PMC
theta, log_w = ais_lib.pmc(gauss_target, dim)

# Convert to standard weights using LSE and normalize
w = np.exp(log_w-np.max(log_w))
w = w/np.sum(w)

# Determine mean of the target distribution
mu_est = np.average(theta, axis=0, weights=w)
print(mu_est)