import numpy as np
from scipy.stats import multivariate_normal as mvn

def pmc(log_target, d, D=100, N=1, I=200, bounds=(-10,10)):
    """
    Runs the population Monte Carlo algorithm
    :param log_target: Logarithm of the target distribution
    :param d: Dimension of the sampling space
    :param D: Number of proposals
    :param N: Number of samples per proposal
    :param I: Number of iterations
    :param bounds: Prior to generate location parameters over [bounds]**d hypercube
    :return particles, weights, and estimate of normalizing constant
    """
    # Initialize the mean of each proposal distribution
    parents = np.random.uniform(bounds[0], bounds[1], (D, d))
    mu = np.repeat(parents, N, axis=0)

    # Initialize the covariance matrix
    sig = np.eye(d)

    # Initialize storage of particles and log weights
    particles = np.zeros((D*N*I, d))
    log_weights = np.ones(D*N*I)*(-np.inf)

    # Initialize start counter
    start=0

    # Loop for the algorithm
    for i in range(I):
        # Update start counter
        stop = start + D*N

        # Generate children particles
        children = mu + (np.matmul(sig, np.random.randn(D*N, d).T)).T
        particles[start:stop, :] = children

        # Compute log proposal
        log_prop = np.array([mvn.logpdf(children[m, :], mean=mu[m, :], cov=sig) for m in range(stop-start)])

        # Compute log weights and store
        log_w = log_target(children) - log_prop
        log_weights[start:stop] = log_w

        # Convert log weights to standard weights using LSE and normalize
        w = np.exp(log_w - np.max(log_w))
        w = w/np.sum(w)

        # Resampling to obtain new parents
        idx = np.random.choice(D*N, D, replace=True, p=w)
        parents = children[idx, :]
        mu = np.repeat(parents, N, axis=0)

        # Update the start index
        start = stop

    return particles, log_weights
