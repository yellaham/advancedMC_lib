import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import ais_lib

# Define target distribution
dim = 2
gauss_target = lambda x: sp.multivariate_normal.logpdf(x, mean=np.zeros(dim), cov=np.eye(dim))

# Run PMC
theta, log_w = ais_lib.pmc(gauss_target, dim)

# Convert to standard weights using LSE and normalize
w = np.exp(log_w-np.max(log_w))
w = w/np.sum(w)

# Sampling importance resampling (SIR) to get approximate posterior samples
idx = np.random.choice(theta.shape[0], 1000, replace=True, p=w)
post_samples = theta[idx, :]

# Plot the approximated target using a weighted histogram
fig1 = plt.hist2d(post_samples[:,0], post_samples[:,1], bins=15)
plt.show()

# D etermine mean of the target distribution
mu_est = np.average(theta, axis=0, weights=w)
print(mu_est)