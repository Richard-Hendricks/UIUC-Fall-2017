import numpy as np
def sigm(z):
    """
    Computes the sigmoid function

    :type z: float
    :rtype: float
    """
    return 1./(1. + np.exp(-z))

def compute_grad(w, x, y):
    """
    Computes gradient of LL for logistic regression

    :type w: 1D np array of weights
    :type x: 2D np array of features where len(w) == len(x[0])
    :type y: 1D np array of labels where len(x) == len(y)
    :rtype: 1D numpy array
    """
    wx = np.dot(x, w)
    sigmwx = np.fromiter((sigm(wxi) for wxi in wx), wx.dtype)
    return np.sum(np.multiply(x, (y - sigmwx).reshape((len(y), 1))), 0)

def gd_single_epoch(w, x, y, step):
    """
    Updates the weight vector by processing the entire training data once

    :type w: 1D numpy array of weights
    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: 1D numpy array of weights
    """
    return w + step*compute_grad(w, x, y)

def gd(x, y, stepsize):
    """
    Iteratively optimizes the objective function by first
    initializing the weight vector with zeros and then
    iteratively updates the weight vector by going through
    the trianing data num_epoch_for_train(global var) times

    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :type stepsize: float
    :rtype: 1D numpy array of weights
    """
    w = np.zeros(len(x[0]))
    for i in range(num_epoch_for_train):
        w = gd_single_epoch(w, x, y, stepsize)
    return w

def predict(w, x):
    """
    Makes a binary decision {0,1} based the weight vector
    and the input features

    :type w: 1D numpy array of weights
    :type x: 1D numpy array of features of a single data point
    :rtype: integer {0,1}
    """
    if sigm(np.dot(x, w)) > 0.5:
        return 1
    return 0

def accuracy(w, x, y):
    """
    Calculates the proportion of correctly predicted results to the total

    :type w: 1D numpy array of weights
    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: float as a proportion of correct labels to the total
    """
    prediction = np.fromiter((predict(w, xi) for xi in x), x.dtype)
    return float(np.sum(np.equal(prediction, y)))/len(y)

def five_fold_cross_validation_avg_accuracy(x, y, stepsize):
    """
    Measures the 5 fold cross validation average accuracy
    Partition the data into five equal size sets like
    |-----|-----|-----|-----|
    For all 5 choose 1 permutations, train on 4, test on 1.

    Compute the average accuracy using the accuracy function
    you wrote.

    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :type stepsize: float
    :rtype: float as average accuracy across the 5 folds
    """
    x_folds = np.split(x, 5)
    y_folds = np.split(y, 5)
    percent_acc_ith_test_fold = np.zeros(5)
    for i in range(5):
        test_index = (i-1)%5
        x_test_fold = x_folds[test_index]
        y_test_fold = y_folds[test_index]
        x_train_folds = np.array([]).reshape((0,len(x[0])))
        y_train_folds = np.array([])
        for j in range(5-1):
            train_index = (i+j)%5
            x_train_folds = np.concatenate((x_train_folds, x_folds[train_index]), axis=0)
            y_train_folds = np.concatenate((y_train_folds, y_folds[train_index]), axis=0)
        w = gd(x_train_folds, y_train_folds, stepsize)
        percent_acc_ith_test_fold[i] = accuracy(w, x_test_fold, y_test_fold)
    return np.average(percent_acc_ith_test_fold)

def tune(x, y):
    """
    Optimizes the stepsize by calculating five_fold_cross_validation_avg_accuracy
    with 10 different stepsizes from 0.001, 0.002,...,0.01 in intervals of 0.001 and
    output the stepsize with the highest accuracy

    For comparison:
    If two accuracies are equal, pick the lower stepsize.

    NOTE: For best practices, we should be using Nested Cross-Validation for
    hyper-parameter search. Without Nested Cross-Validation, we bias the model to the
    data. We will not implement nested cross-validation for now. You can experiment with
    it yourself.
    See: http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: float as best stepsize
    """
    stepsizes = np.linspace(0.001, 0.01, num=10)
    acc = np.zeros(10)
    for (i, step) in enumerate(stepsizes):
        acc[i] = five_fold_cross_validation_avg_accuracy(x_train, y_train, step)
    return stepsizes[np.argmax(acc)]

# Top three tuned stepsize values
def tune_top_three(x, y):
    stepsizes = np.linspace(0.001, 0.01, num=10)
    acc = np.zeros(10)
    for (i, step) in enumerate(stepsizes):
        acc[i] = five_fold_cross_validation_avg_accuracy(x_train, y_train, step)
    return stepsizes[np.argpartition(acc, -3)[-3:]]


# Correct Results
w_single_epoch = gd_single_epoch(np.zeros(len(x_train[0])), x_train, y_train, default_stepsize)
w_optimized = gd(x_train, y_train, default_stepsize)
y_predictions = np.fromiter((predict(w_optimized, xi) for xi in x_test), x_test.dtype)
five_fold_average_accuracy = five_fold_cross_validation_avg_accuracy(x_train, y_train, default_stepsize)
# tuned_stepsize = tune(x_train, y_train)
top_three_tuned_stepsizes = tune_top_three(x_train, y_train)
