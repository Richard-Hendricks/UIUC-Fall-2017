import numpy as np
# http://www.cs.cmu.edu/~tom/10601_sp08/slides/recitation-mle-nb.pdf
# http://www.cs.columbia.edu/~mcollins/em.pdf
def MLE(data, labels):
  """
   Please follow the format of fA_params and fA_pi, and return the
   result in the format of (fA_params,fA_pi)
  :type data: numpy array
  :type labels: numpy array
  :rtype: tuple
  """
  # Number of label occurrences
  counts = np.unique(labels, return_counts = True)[1]
  pi = counts / counts.sum()
  
  # Indices is a list, each element is the index set for each label (3 sublists)
  indices = []
  for label in np.unique(labels, return_counts = True)[0]:
      indices.append([index for index, x in enumerate(list(labels)) if x == label])

  # Split training data into three groups by label
  # training_data[indices[0]] label = 0
  params = fA_params.copy()
  for i in range(3):
      params[0][i] = np.mean(data[indices[i]], axis = 0) # MU estimation
      params[1][i] = np.mean((data[indices[i]] - params[0][i]) ** 2, 
            axis = 0) # Sigma estimation

  return (params, pi)

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
