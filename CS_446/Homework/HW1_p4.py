import numpy as np
# https://beckernick.github.io/logistic-regression-from-scratch/
def sigm(z):
    """
    Computes the sigmoid function

    :type z: float
    :rtype: float
    """
    return (1 / (1 + np.exp(-z)))

def compute_grad(w, x, y):
    """
    Computes gradient of LL for logistic regression

    :type w: 1D np array of weights
    :type x: 2D np array of features where len(w) == len(x[0])
    :type y: 1D np array of labels where len(x) == len(y)
    :rtype: 1D numpy array
    """
    scores = np.dot(x, w)
    predictions = sigm(scores)
    gradient = np.dot(x.T, y - predictions)
    
    return (gradient)

def gd_single_epoch(w, x, y, step):
    """
    Updates the weight vector by processing the entire training data once

    :type w: 1D numpy array of weights
    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: 1D numpy array of weights
    """
    new_weights = w + step * compute_grad(w, x, y)
    
    return (new_weights)

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
    # Initialize weights
    weights = np.zeros(len(x[0]))
    
    for i in range(num_epoch_for_train):
        weights = gd_single_epoch(weights, x, y, stepsize)
        
    return(weights)

def predict(w, x):
    """
    Makes a binary decision {0,1} based the weight vector
    and the input features

    :type w: 1D numpy array of weights
    :type x: 1D numpy array of features of a single data point
    :rtype: integer {0,1}
    """
    # Use the final weights to get the logits for dataset
    final_scores = np.dot(x, w)
    
    # Round them to the nearest integer (0 or 1)
    preds = np.round(sigm(final_scores))
    
    return (preds)

def accuracy(w, x, y):
    """
    Calculates the proportion of correctly predicted results to the total

    :type w: 1D numpy array of weights
    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: float as a proportion of correct labels to the total
    """
    preds = predict(w, x) # Predictions
    acc = (preds == y).sum().astype(float) / len(preds)
    
    return (acc)

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
    
    # Split the dataset into 5 folds
    x_split = list()
    y_split = list()
    x_copy = list(x) # Change back to x
    y_copy = list(y)
    fold_size = int(len(x) / 5)
    for i in range(5):
        x_fold = list()
        y_fold = list()
        while len(x_fold) < fold_size:
            x_fold.append(x_copy.pop(0))
            y_fold.append(y_copy.pop(0))
        x_split.append(x_fold)
        y_split.append(y_fold)
    
    # Train model
    acc_list = list()
    
    for i in range(5):
        x_split_copy = x_split.copy() # REMEMBER to use .copy()!!!
        y_split_copy = y_split.copy()
        x_tst = np.array(x_split_copy)[i]
        y_tst = np.array(y_split_copy)[i]
        del x_split_copy[i]
        del y_split_copy[i]
        x_trn = np.vstack(np.array(x_split_copy))
        y_trn = np.ndarray.flatten(np.vstack(np.array(y_split_copy)))
        
        weights = gd(x_trn, y_trn, stepsize)
        acc_list.append(accuracy(weights, x_tst, y_tst))
    
    return (np.mean(np.array(acc_list)))
        
    

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
    acc_list = [] # Initialize accuracy list
    
    for step in np.arange(0.001, 0.011, 0.001):
        acc_list.append(five_fold_cross_validation_avg_accuracy(x, y, step))
    
    index = acc_list.index(max(acc_list))
    
    return (np.arange(0.001, 0.011, 0.001)[index])
    
    
w_single_epoch = gd_single_epoch(np.zeros(len(x_train[0])), x_train, y_train, default_stepsize)

w_optimized = gd(x_train, y_train, default_stepsize)

y_predictions = np.fromiter((predict(w_optimized, xi) for xi in x_test), x_test.dtype)

five_fold_average_accuracy = five_fold_cross_validation_avg_accuracy(x_train, y_train, default_stepsize)

tuned_stepsize = tune(x_train, y_train)
    
    
    
    
    
    
