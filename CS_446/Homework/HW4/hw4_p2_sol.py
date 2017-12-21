import numpy as np
import numpy.linalg as la
from scipy.stats import multivariate_normal
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

    weights = np.zeros((n, 2))

    for z in range(num_iterations):
        for i in range(n):
            pdf_1 = multivariate_normal.pdf(data[i, :], mean=mu_1, cov=sigma_1)
            pdf_2 = multivariate_normal.pdf(data[i, :], mean=mu_2, cov=sigma_2)
            weights[i, 0] = (pi_1 * pdf_1) / (pi_1 * pdf_1 + pi_2 * pdf_2)
        weights[:, 1] = 1 - weights[:, 0]

        N1 = np.sum(weights[:, 0])
        N2 = np.sum(weights[:, 1])

        pi_1 = N1 / n
        pi_2 = N2 / n

        mu_1 = np.sum(weights[:, 0:1] * data, axis=0) / N1
        mu_2 = np.sum(weights[:, 1:2] * data, axis=0) / N2

        sigma_1 = np.cov(m=data.T, aweights=weights[:, 0], ddof=0)
        sigma_2 = np.cov(m=data.T, aweights=weights[:, 1], ddof=0)

    return pi_1, pi_2, mu_1, mu_2, sigma_1, sigma_2


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

    weights = np.zeros((n, 2))

    for z in range(num_iterations):
        weights[:, 0] = np.sign(la.norm(data - np.expand_dims(centroid_1, axis=0), axis=-1) -
                                la.norm(data - np.expand_dims(centroid_2, axis=0), axis=-1)) / 2 + 0.5
        weights[:, 1] = 1 - weights[:, 0]

        centroid_1 = np.mean(weights[:, 0:1] * data, axis=0)
        centroid_2 = np.mean(weights[:, 1:2] * data, axis=0)

    return centroid_1, centroid_2

# Run the GMM
pi_1, pi_2, mu_1, mu_2, sigma_1, sigma_2 = gaussian_mixture_model(input_data)

# Run K-means
centroid_1, centroid_2 = kmeans(input_data)

# Make the hard cluster assignments and plot the data

# Part 1: plot the MoG assignments
colors = []
for i in range(input_data.shape[0]):
    pdf_1 = multivariate_normal.pdf(input_data[i, :], mean=mu_1, cov=sigma_1)
    pdf_2 = multivariate_normal.pdf(input_data[i, :], mean=mu_2, cov=sigma_2)
    colors.append("r" if pi_1 * pdf_1 > pi_2 * pdf_2 else "b")
plt.figure()
plt.scatter(input_data[:, 0], input_data[:, 1], c=colors)
plt.title("Cluster Assignments (MoG)")
plt.show()

# Part 2: plot the K-means assignments
colors = []
for i in range(input_data.shape[0]):
    weight_1 = 1 / la.norm(input_data[i, :] - centroid_1)
    weight_2 = 1 / la.norm(input_data[i, :] - centroid_2)
    colors.append("r" if weight_1 > weight_2 else "b")
plt.figure()
plt.scatter(input_data[:, 0], input_data[:, 1], c=colors)
plt.title("Cluster Assignments (K-means)")
plt.show()
