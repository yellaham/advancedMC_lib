import numpy as np

def pmc(log_target, d, D=500, N=1, I=400, bounds=[-10,10]):
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

