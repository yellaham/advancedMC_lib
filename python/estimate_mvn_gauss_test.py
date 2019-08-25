import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import ais_lib as ais

# Define target distribution
dim = 10
mu_pi = np.ones(dim)
sigma_pi = np.eye(dim)
gauss_target = lambda x: sp.multivariate_normal.logpdf(x, mean=mu_pi, cov=sigma_pi)

# Run PMC
result_pmc = ais.pmc(gauss_target, dim, weighting_scheme='DM')
theta = result_pmc.particles
log_w = result_pmc.log_weights

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
print('Estimated mean: ', mu_est)

# Compute the MSE
MSE = (np.linalg.norm(mu_est-mu_pi)**2)/dim
print('MSE: ',MSE)