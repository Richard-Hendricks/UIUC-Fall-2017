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

MLE_result = MLE(training_data, training_label)

def apply(data,params, pi):
    """
    :type data: 1D numpy array
    :type params: 2D numpy array for mu and sigma^2
    :type pi: 1D numpy array for pi
    :rtype: 1D numpy array, the normalized predicted distribution
    """
    pi_0 = pi[0] * np.prod(1 / (np.sqrt(2 * np.pi * params[1][0])) * np.exp( - (data - params[0][0]) ** 2 / (2 * params[1][0])))
    pi_1 = pi[1] * np.prod(1 / (np.sqrt(2 * np.pi * params[1][1])) * np.exp( - (data - params[0][1]) ** 2 / (2 * params[1][1])))
    pi_2 = pi[2] * np.prod(1 / (np.sqrt(2 * np.pi * params[1][2])) * np.exp( - (data - params[0][2]) ** 2 / (2 * params[1][2])))

    return (np.array([pi_0, pi_1, pi_2]) / (pi_0 + pi_1 + pi_2))

predicted_distr = np.random.rand(5, 3)
for i in range(5):
    predicted_distr[i] = apply(training_data[i], MLE_result[0], MLE_result[1])

  
# https://www.cs.utah.edu/~piyush/teaching/20-9-print.pdf
# http://www.cs.cmu.edu/~aarti/Class/10601/slides/GNB_LR_9-29-2011.pdf
def MAP(data,prior_params, prior_pi):
    """
    :type data: 2D numpy array
    :type params: 2D numpy array for mu and sigma^2
    :type pi: 1D numpy array for pi, recall that this is fA_pi[1,:]
    :rtype: 1D numpy array, the normalized predicted distribution
    """

    # Indices is a list, each element is the index set for each label (3 sublists)
    indices = []
    for label in np.unique(training_label, return_counts = True)[0]:
        indices.append([index for index, x in enumerate(list(training_label)) if x == label])

    
    # Split training data into three groups by label
    # training_data[indices[0]] label = 0
    params = prior_params.copy()
    sample_vars = np.array([])
    
    for i in range(3):
        sample_vars[i] = np.var(data[indices[i]])
        params[0][i] = (sample_vars[i] * prior_params[0][i] + prior_params[1][i] * data[indices[i]].sum(axis = 0)) / (counts[i] * prior_params[1][i] + sample_vars[i])# MU estimation
        params[1][i] = np.mean((data[indices[i]] - params[0][i]) ** 2, axis = 0) # Sigma estimation
        
    counts = np.unique(training_label, return_counts = True)[1]
    class_prob = (counts + prior_pi - np.ones(3)) / (data.shape[0] + prior_pi.sum() - np.ones(3) * 3)


    return (params, class_prob)




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
