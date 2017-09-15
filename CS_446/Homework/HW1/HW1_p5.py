import numpy as np
def MLE(data, labels):
  """
   Please follow the format of fA_params and fA_pi, and return the
   result in the format of (fA_params,fA_pi)
  :type data: numpy array
  :type labels: numpy array
  :rtype: tuple
  """

def apply(data,params, pi):
  """
  :type data: 1D numpy array
  :type params: 2D numpy array for mu and sigma^2
  :type pi: 1D numpy array for pi
  :rtype: 1D numpy array, the normalized predicted distribution
  """

def MAP(data,prior_params, prior_pi):
  """
  :type data: 2D numpy array
  :type params: 2D numpy array for mu and sigma^2
  :type pi: 1D numpy array for pi, recall that this is fA_pi[2,:]
  :rtype: 1D numpy array, the normalized predicted distribution
  """

def CV(training_data, training_label, prior_params, prior_pi, k):
    """
    k_fold_cross_validation_avg_accuracy measures the k fold cross validation
    average accuracy
    :type training_data: 2D numpy array of features
    :type training_label: 1D numpy array of labels
    :type prior_params: parameter set of mu and sigma^2, as prior
    :type prior_pi: parameter set of pi, as prior
    :type k: integer of number of folds
    :rtype: float as average accuracy across the k folds
    """
