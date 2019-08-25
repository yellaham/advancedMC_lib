import numpy as np
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm

def vectorized_gaussian_logpdf(X, means, covariances):
    _, d = X.shape
    constant = d * np.log(2 * np.pi)
    log_determinants = np.log(np.prod(covariances, axis=1))
    deviations = X - means
    inverses = 1 / covariances
    return -0.5 * (constant + log_determinants +
        np.sum(deviations * inverses * deviations, axis=1))

class AIS_sampler:
    def __init__(self, X, log_W):
        self.particles = X
        self.log_weights = log_W

def pmc(log_target, d, D=100, N=1, I=200, var_prop=1, bounds=(-10,10), weighting_scheme="standard"):
    """
    Runs the population Monte Carlo algorithm
    :param log_target: Logarithm of the target distribution
    :param d: Dimension of the sampling space
    :param D: Number of proposals
    :param N: Number of samples per proposal
    :param I: Number of iterations
    :param var_prop: Variance of each proposal distribution
    :param bounds: Prior to generate location parameters over [bounds]**d hypercube
    :return particles, weights, and estimate of normalizing constant
    """
    # Initialize the mean of each proposal distribution
    parents = np.random.uniform(bounds[0], bounds[1], (D, d))
    mu = np.repeat(parents, N, axis=0)

    # Initialize the covariance matrix
    sig = var_prop*np.eye(d)

    # Initialize storage of particles and log weights
    particles = np.zeros((D*N*I, d))
    log_weights = np.ones(D*N*I)*(-np.inf)

    # Initialize start counter
    start=0

    # Loop for the algorithm
    for i in tqdm(range(I)):
        # Update start counter
        stop = start + D*N

        # Generate children particles
        children = mu + (np.matmul(sig, np.random.randn(D*N, d).T)).T
        particles[start:stop, :] = children

        # Compute log proposal
        if weighting_scheme == 'standard':
            log_prop = vectorized_gaussian_logpdf(children, mu, var_prop*np.ones((D*N,d)))
            #log_prop = np.array([mvn.logpdf(children[m, :], mean=mu[m, :], cov=sig) for m in range(stop-start)])
        elif weighting_scheme == 'DM':
            prop = np.zeros(D*N)
            for j in range(D):
                prop += (1/D)*mvn.pdf(children, mean=mu[j, :], cov=sig)
            log_prop = np.log(prop)
        else:
            log_prop = vectorized_gaussian_logpdf(children, mu, var_prop*np.ones((D*N,d)))
            #log_prop = np.array([mvn.logpdf(children[m, :], mean=mu[m, :], cov=sig) for m in range(stop-start)])

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

    # Generate output
    output = AIS_sampler(particles, log_weights)

    return output
