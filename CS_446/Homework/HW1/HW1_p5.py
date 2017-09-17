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
    localcounts = np.unique(training_label, return_counts = True)[1]
    indices = []
    for label in np.unique(training_label, return_counts = True)[0]:
        indices.append([index for index, x in enumerate(list(training_label)) if x == label])

    
    # Split training data into three groups by label
    # training_data[indices[0]] label = 0
    params = prior_params.copy()
    sample_vars = np.ones(3)
    
    for i in range(3):
        sample_vars[i] = np.var(data[indices[i]])
        params[0][i] = (sample_vars[i] * prior_params[0][i] + prior_params[1][i] * data[indices[i]].sum(axis = 0)) / (localcounts[i] * prior_params[1][i] + sample_vars[i])# MU estimation
        params[1][i] = np.mean((data[indices[i]] - params[0][i]) ** 2, axis = 0) # Sigma estimation
     
    
    class_prob = (localcounts + prior_pi - 1) / (data.shape[0] + prior_pi.sum() - 3)


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
      
      
    def pred_class(training): # return a 1D numpy array of labels
        pred_class = np.ones(training.shape[0])
         
        for i in range(training.shape[0]):
            pred_class[i] = np.argmax(apply(training[0], prior_params, prior_pi[0, :]))
            
        return (pred_class)
      
    def accuracy(pred_class, training_label):
        return ((pred_class == training_label).sum().astype(float) / len(pred_class))
    
        # Split the dataset into 5 folds
    x_split = list()
    y_split = list()
    x_copy = list(training_data) # Change back to x
    y_copy = list(training_label)
    fold_size = int(len(training_data) / 5)
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
        

        acc_list.append(accuracy(pred_class(x_trn), y_trn))
        
        
    return (np.mean(np.array(acc_list)))
    

index = np.argmax(np.array([CV(training_data, training_label, fA_params, fA_pi, 3),
                            CV(training_data, training_label, fB_params, fB_pi, 3),
                            CV(training_data, training_label, fC_params, fC_pi, 3)]))
      
if index == 0:
    best_prior = (fA_params, fA_pi)
elif index == 1:
    best_prior = (fB_params, fB_pi)
else:
    best_prior = (fC_params, fC_pi)
    
    
    
MAP_params = MAP(training_data, best_prior[0], best_prior[1][1, :])  

predictions = np.ones(testing_data.shape[0])         
for i in range(testing_data.shape[0]):
    probs0 = np.prod(1 / (np.sqrt(2 * np.pi * MAP_params[0][1][0])) * np.exp(-(testing_data[i] - MAP_params[0][0][0]) ** 2 / (2 * MAP_params[0][1][0]))) * MAP_params[1][0]
    probs1 = np.prod(1 / (np.sqrt(2 * np.pi * MAP_params[0][1][1])) * np.exp(-(testing_data[i] - MAP_params[0][0][1]) ** 2 / (2 * MAP_params[0][1][1]))) * MAP_params[1][1]
    probs2 = np.prod(1 / (np.sqrt(2 * np.pi * MAP_params[0][1][2])) * np.exp(-(testing_data[i] - MAP_params[0][0][2]) ** 2 / (2 * MAP_params[0][1][2]))) * MAP_params[1][2]
    
    predictions[i] = np.argmax(np.array([probs0, probs1, probs2]))
            
 
    
    
      
