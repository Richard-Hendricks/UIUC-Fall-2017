import numpy as np

# Reference: http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex8/ex8.html
def rbf_kernel(x1, x2, gamma=100):
    """
    RBF Kernel

    :type x1: 1D numpy array of features
    :type x2: 1D numpy array of features
    :type gamma: float bandwidth
    :rtype: float
    """
    return np.exp(-gamma * ((x1 - x2) ** 2).sum())
    # Student implementation here

def w_dot_x(x, x_list, y_list, alpha, kernel_func=rbf_kernel):
    """
    Calculates wx using the training data

    :type x: 1D numpy array of features
    :type x_list: 2D numpy array of training features
    :type y_list: 1D numpy array of training labels where len(x) == len(y)
    :type alpha: 1D numpy array of int counts
    :type kernel_func: function that takes in 2 vectors and returns a float
    :rtype: float representing wx
    """
    # Student implementation here
    # This is to ensure that it is always > 0 when we divide
    norm_const = max(1, np.sum(alpha))

def predict(x, x_list, y_list, alpha):
    """
    Predicts the label {-1,1} for a given x

    :type x_list: 2D numpy array of training features
    :type y_list: 1D numpy array of training labels where len(x) == len(y)
    :rtype: Integer in {-1,1}
    """
    # Student implementation here

def pegasos_train(x_list, y_list, tol=0.01):
    """
    Trains a predictor using the Kernel Pegasos Algorithm
    Use the tol metric to trigger the update

    :type x_list: 2D numpy array of training features
    :type y_list: 1D numpy array of training labels where len(x) == len(y)
    :rtype: 1D numpy array of int counts
    """
    # Alpha counts the number of times the traning example has been selected
    alpha = np.zeros(len(x_list), dtype=int)
    for t in range(num_samples_to_train):
        # Random index to pick the sample for SGD
        rand_i = np.random.randint(len(x_list))
        # Student implementation here
    return alpha

def accuracy(x_list, y_list, alpha, x_test, y_test):
    """
    Calculates the proportion of correctly predicted results to the total

    :type x_list: 2D numpy array of training features
    :type y_list: 1D numpy array of training labels where len(x) == len(y)
    :type alpha: 1D numpy array of int counts
    :type x_test: 2D numpy array of test features
    :type y_test: 1D numpy array of test labels where len(x) == len(y)
    :rtype: float as a proportion of correct labels to the total
    """
    prediction = np.fromiter((predict(xi, x_list, y_list, alpha) for xi in x_test), x_test.dtype)
    return float(np.sum(np.equal(prediction, y_test)))/len(y_test)

# You may use this
# final_alpha = pegasos_train(x_train, y_train)
# print("Total Accuracy is: " + str(accuracy(x_train, y_train, final_alpha, x_test, y_test)))
