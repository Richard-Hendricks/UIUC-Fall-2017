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
    X[:, -1] = np.ones(X.shape[0]) # add intercept
    
    n = data.shape[0]
    m = data.shape[1]
    y = data[:, m - 1]
    # J = np.sum(loss ** 2) / (2 * m) cost function
    gradient = 2 * np.dot(y - np.dot(X, weights), -X) / n
    # gradient = np.dot(-X.T, data[:, m - 1] - np.dot(X, weights))
    
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
    w = np.zeros(data.shape[1]) # Initialize weights
    
    for i in range(iterations):
        w = w - alpha * gradient(data, w)
    
    return w
    pass


# Step 3: Standardize the features (but not the labels)
X = np.delete(golf_data, np.s_[-1: ], axis = 1)
X_mean = X.mean(axis = 0)
X_std = X.std(axis = 0)
golf_data_standardized = golf_data.copy()  # Implement me!

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i, j] = (X[i, j] - X_mean[j])/ X_std[j]

golf_data_standardized[:, :-1] = X.copy()

# Step 4: Implement Forward Selection
def forward_selection(data, max_var):
    #https://github.com/shiluyuan/forward-variable-selection-in-python/blob/html/Forward_selection.ipynb
    """
    Computes the top max_var number of features by forward selection

    :param numpy.ndarray data: numpy.ndarray data: A (n, m) numpy array, where n = number of examples and
    m=(features + 1). The last column is the label y for that example
    :type max_var: integer
    :returns A (max_var,) numpy array whose values are the features that were selected
    :rtype: numpy.ndarray
    """
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
    X = data.copy()
    X[:, -1] = np.ones(X.shape[0]) # add intercept
    y = data[:, -1].copy() # label

    k = 1 # current number of variables in the model
    indices = [] # stores the selected feature indices
    
    while k <= max_var:
        RSS = []
        for i in range(X.shape[1] - 1):
            # iterate over features
            if i not in indices:
                
                new_data = np.squeeze(data[:, [indices + [i] + [data.shape[1] - 1]]])# data will by order
                params, intercept = gradient_descent(new_data, 0.1, 200)[:- 1], gradient_descent(new_data, 0.1, 200)[-1]
                pred = np.dot(X[:, indices + [i]], params) + intercept
                RSS.append(np.square(y - pred).sum())
        
        # Insert zeros to the original position
        for index in sorted(indices):
            RSS.insert(index, 0)
            
        indices.append(np.argsort(np.array(RSS))[len(indices)])
        k = k + 1
        
    return np.array(indices)
    
 #columns[np.argsort(scores)][::-1]
    pass

forward_result = forward_selection(golf_data_standardized, 5)  # Implement me!

k = 1
while k <= 5:
    print(k)
    k = k+1



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
    X = data.copy()
    X[:, -1] = np.ones(X.shape[0]) # add intercept
    y = data[:, -1].copy() # label

    k = 1 # current number of variables in the model
    indices = list(range(X.shape[1] - 1)) # stores the selected feature indices
    
    while k <= (X.shape[1] - 1 - max_var):
        RSS = []
        for i in range(X.shape[1] - 1):
            # iterate over features
            if i in indices:
                temp = indices.copy()
                new_data = np.squeeze(data[:, [[item for item in temp if item != i] + [data.shape[1] - 1]]])# data will by order
                params, intercept = gradient_descent(new_data, 0.1, 200)[:- 1], gradient_descent(new_data, 0.1, 200)[-1]
                pred = np.dot(X[:, [item for item in temp if item != i]], params) + intercept
                RSS.append(np.square(y - pred).sum())
        
        # Insert zeros to the original position
        for index in sorted(list(set(list(range(X.shape[1] - 1))) - set(indices))):
            RSS.insert(index, 0)
            
        indices.remove(np.argsort(np.array(RSS))[X.shape[1] - 1 - len(indices)])
        k = k + 1
    
    return np.array(indices)
    pass


backward_result = backward_elimination(golf_data_standardized, 5)


def soft_threshold(lambda0, w):
    """
    W is an array
    """
    result = w.copy()
    
    for i in range(len(w)):
        if w[i] > lambda0:
            result[i] = w[i] - lambda0
        elif w[i] < -lambda0:
            result[i] = w[i] + lambda0
        else:
            result[i] = 0
            
    return result


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
    # http://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/08-prox-grad.pdf
    # Implement me!
    w = np.zeros(data.shape[1]) # Initialize weights
    X = data.copy()
    X[:, -1] = np.ones(X.shape[0]) # add intercept
    
    n = data.shape[0]
    m = data.shape[1]
    y = data[:, m - 1]
    
    for i in range(iterations):
        std_gradient = w - alpha * gradient(data, w)
        w = np.append(soft_threshold(penalty, std_gradient[:-1]), std_gradient[-1])
    
    return w
    pass
