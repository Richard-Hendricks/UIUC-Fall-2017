import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def gaussian_mixture_model(data):
    """
    Runs the EM algorithm for a Guassian mixture model with 2 gaussians on the given dataset, and returns the
    parameters for the model.
    :param np.ndarray data: The data of size (n, k), where n is the number of points and k is the dimensionality of
    each point.
    :rtype tuple: (pi_1, pi_2, mu_1, mu_2, sigma_1, sigma_2)
    """

    n = data.shape[0]  # number of points
    k = data.shape[1]  # the dimensionality of each point
    num_iterations = 80

    # initialize the priors to be equal
    pi_1 = 0.5
    pi_2 = 0.5

    # get the initial means as random datapoints
    initial_means = rnd.permutation(data)[:2]
    mu_1 = initial_means[0, :]
    mu_2 = initial_means[1, :]

    # start with the covariance matrices as the identity matrix times a large constant
    sigma_1 = np.identity(k) * 100
    sigma_2 = np.identity(k) * 100

    # Implement me!
    return None, None, None, None, None, None

def kmeans(data):
    """
    Runs the K-means algorithm to find two clusters on the given dataset, and returns the centroids for each cluster.
    :param np.ndarray data: The data of size (n, k), where n is the number of points and k is the dimensionality of
    each point.
    :rtype tuple: (centroid_1, centroid_2)
    """

    n = data.shape[0]  # number of points
    num_iterations = 80

    # get the initial means as random datapoints
    initial_means = rnd.permutation(data)[:2]
    centroid_1 = initial_means[0, :]
    centroid_2 = initial_means[1, :]

    # Implement me!
    return None, None

# Run the GMM
pi_1, pi_2, mu_1, mu_2, sigma_1, sigma_2 = gaussian_mixture_model(input_data)

# Run K-means
centroid_1, centroid_2 = kmeans(input_data)

# Make the hard cluster assignments and plot the data
# Implement me!
