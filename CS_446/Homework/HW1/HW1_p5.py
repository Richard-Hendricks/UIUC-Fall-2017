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
    :type params: 3D numpy array for mu and sigma^2
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
def MAP(data, labels, prior_params, pseudo_count):
    """
    :type data: 2D numpy array
    :type labels: 1D numpy array
    :type params: 3D numpy array for mu and sigma^2
    :type pseudo_count: 1D numpy array for pi, recall that this is fA_pi[1,:]
    :rtype:tuple, same format as MLE
    """

    # Indices is a list, each element is the index set for each label (3 sublists)
    localcounts = np.unique(labels, return_counts = True)[1]
    indices = []
    for label in np.unique(labels, return_counts = True)[0]:
        indices.append([index for index, x in enumerate(list(labels)) if x == label])

    
    # Split training data into three groups by label
    # training_data[indices[0]] label = 0
    params = prior_params.copy()
    sample_vars = []
    
    for i in range(3):
        sample_vars.append(np.var(data[indices[i]], axis = 0))
        params[0][i] = (sample_vars[i] * prior_params[0][i] + prior_params[1][i] * data[indices[i]].sum(axis = 0)) / (localcounts[i] * prior_params[1][i] + sample_vars[i])# MU estimation
        params[1][i] = sample_vars[i] # Sigma estimation
       
    class_prob = (localcounts + pseudo_count - 1) / (data.shape[0] + pseudo_count.sum() - 3)

    return (params, class_prob)

MAP_result = MAP(training_data, training_label, fA_params, fA_pi[1,: ])

def accuracy(predictions, actual):
    return ((predictions == actual).sum().astype(float) / len(actual))


def pred_class(training_set, prior_params, prior_pi):
    pred_class = np.ones(training_set.shape[0])
    
    for i in range(training_set.shape[0]):
        # Return the index of maximum value in the array
        pred_class[i] = np.argmax(apply(training_set[i], prior_params, prior_pi[0, :]))
    
    return (pred_class)
    

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
    
    # Split the dataset into 5 folds
    fold_size = int((1 / k) * training_data.shape[0])
    acc = np.ones(k)
    for i in range(k):
        tst_idx = range(i * fold_size, (i + 1) * fold_size)
        trn_idx = [trn_idx for trn_idx in range(training_data.shape[0]) if trn_idx not in tst_idx]
        x_trn = training_data[trn_idx, ]
        x_tst = training_data[tst_idx, ]
        y_trn = training_label[trn_idx, ]
        y_tst = training_label[tst_idx, ]
        para = MAP(x_trn, y_trn, prior_params, prior_pi[1,:])
        
        w = np.random.rand(x_tst.shape[0], 3)
        for j in range(x_tst.shape[0]):
            w[j] = apply(x_tst[j], para[0], para[1])

        best_result = np.argsort(w)[:,len(para[1])-1] + 1
        acc[i] = np.mean(y_tst == best_result)
    
    return np.mean(acc)
        
    
   
fB_params = MLE_result[0]
fB_pi = np.random.rand(2, 3)
fB_pi[0,:] = MLE_result[1]
fB_pi[1,:] = np.unique(training_label, return_counts = True)[1]

index = np.argmax(np.array([CV(training_data, training_label, fA_params, fA_pi, 5),
                            CV(training_data, training_label, fB_params, fB_pi, 5),
                            CV(training_data, training_label, fC_params, fC_pi, 5)]))
      
if index == 0:
    best_prior = (fA_params, fA_pi[0,:])
    MAP_params = MAP(training_data, training_label, fA_params, fA_pi[1,: ])#???
elif index == 1:
    best_prior = (fB_params, fB_pi[0,:])
    MAP_params = MAP(training_data, training_label, fB_params, fB_pi[1,: ])
else:
    best_prior = (fC_params, fC_pi[0,:])
    MAP_params = MAP(training_data, training_label, fC_params, fC_pi[1,: ])
    
    

w = np.random.rand(testing_data.shape[0], 3)
for i in range(testing_data.shape[0]):
# Return the index of maximum value in the array
    w[i] = apply(testing_data[i], MAP_params[0], MAP_params[1])

predictions = np.argsort(w)[:,w.shape[1]-1] + 1       
#for i in range(testing_data.shape[0]):
  #  probs0 = np.prod(1 / (np.sqrt(2 * np.pi * MAP_params[0][1][0])) * np.exp(-(testing_data[i] - MAP_params[0][0][0]) ** 2 / (2 * MAP_params[0][1][0]))) * MAP_params[1][0]
  #  probs1 = np.prod(1 / (np.sqrt(2 * np.pi * MAP_params[0][1][1])) * np.exp(-(testing_data[i] - MAP_params[0][0][1]) ** 2 / (2 * MAP_params[0][1][1]))) * MAP_params[1][1]
   # probs2 = np.prod(1 / (np.sqrt(2 * np.pi * MAP_params[0][1][2])) * np.exp(-(testing_data[i] - MAP_params[0][0][2]) ** 2 / (2 * MAP_params[0][1][2]))) * MAP_params[1][2]
    
   # predictions[i] = np.argmax(np.array([probs0, probs1, probs2]))
            
 
    
    
      
