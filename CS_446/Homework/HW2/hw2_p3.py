#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:55:20 2017

@author: Yiming
"""

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create the plots
# Implement me!

for i in range(golf_data.shape[1] - 1):
    fig, ax = plt.subplots(figsize = (10, 6))
    plt.scatter(golf_data[:, i], golf_data[:, -1])
    plt.title('Feature %s'%i )
    plt.grid(color = 'lightgray', linestyle = '--')
    plt.show()

# Step 2: Define the linear regression function using gradient descent
def gradient(data, weights):
    """
    Computes the gradient of the residual sum-of-squares (RSS) function for the given dataset and current weight values

    :param numpy.ndarray data: A (n, m) numpy array, where n = number of examples and m=(features + 1). The last column
    is the label y
    for that example
    :param numpy.ndarray weights: A (m,) numpy array, where weights[m-1] is the bias, and weights[i] corresponds to the
    weight for
    feature i in data
    :returns A (m,) numpy array, equaling the gradient of RSS at this point.
    :rtype: numpy.ndarray

    """
    # http://charlesfranzen.com/posts/multiple-regression-in-python-gradient-descent/
    # http://www.ozzieliu.com/tutorials/Linear-Regression-Gradient-Descent.html
    # X = np.delete(data, np.s_[-1: ], axis = 1)
    
    X = data.copy()
    X[:, -1] = np.ones(X.shape[0])
    y = data[:, -1] # labels
    
    # m: number of samples
    m = X.shape[0]
    hypothesis = np.dot(X, weights) 
    loss = hypothesis - y
    # J = np.sum(loss ** 2) / (2 * m) cost function
    gradient = np.dot(X.T, loss) / m
    
    return gradient

    # Implement me!
    pass


def gradient_descent(data, alpha, iterations):
    """
    Performs gradient descent using the supplied data, learning rate, and number of iterations. Start with weights =
    the zero vector.

    :param numpy.ndarray data: A (n, m) numpy array, where n = number of examples and m=(features + 1). The last column
    is the label y for that example
    :param float alpha: A real value corresponding to the step size for each descent step
    :param int iterations: The total number of iterations (steps) to take
    :returns A (m,) numpy array of the final weight vector after the gradient descent process
    :rtype: numpy.ndarray
    """

    # Implement me!
    w = np.zeros(len(data[0]) - 1) # Initialize weights
    
    for i in range(iterations):
        w = w - alpha * compute_grad(w, x, y)
    
    pass


# Step 3: Standardize the features (but not the labels)
golf_data_standardized = None  # Implement me!


# Step 4: Implement Forward Selection
def forward_selection(data, max_var):
    """
    Computes the top max_var number of features by forward selection

    :param numpy.ndarray data: numpy.ndarray data: A (n, m) numpy array, where n = number of examples and
    m=(features + 1). The last column is the label y for that example
    :type max_var: integer
    :returns A (max_var,) numpy array whose values are the features that were selected
    :rtype: numpy.ndarray
    """

    # Implement me!
    pass

forward_result = None  # Implement me!


# Step 5: Implement Backward Elimination
def backward_elimination(data, max_var):
    """
    Computes the top max_var number of features by backward elimination

    :param numpy.ndarray data: numpy.ndarray data: A (n, m) numpy array, where n = number of examples and
    m=(features + 1). The last column is the label y for that example
    :type max_var: integer
    :returns A (max_var,) numpy array whose values are the features that were selected
    :rtype: numpy.ndarray
    """

    # Implement me!
    pass


backward_result = None  # Implement me!


# Step 6: Implemnt Gradient Descent with Lasso
def gradient_descent_lasso(data, alpha, iterations, penalty):
    """
    Performs gradient descent using the supplied data, learning rate, number of iterations, and LASSO penalty (lambda).
    The code for this should be structurally the same as gradient_descent, with the exception that after each iteration
    you will pass the weight vector through the LASSO projection. Start with weights = the zero vector.

    :param numpy.ndarray data: A (n, m) numpy array, where n = number of examples and m=(features + 1). The last column
    is the label y for that example
    :param float alpha: A real value corresponding to the step size for each descent step
    :param int iterations: The total number of iterations (steps) to take
    :param float penalty: A real positive value representing the LASSO penalty (lambda)
    :returns A (m,) numpy array of the final weight vector after the LASSO gradient descent process
    :rtype: numpy.ndarray
    """

    # Implement me!
    pass
