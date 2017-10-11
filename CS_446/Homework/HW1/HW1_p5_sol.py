import numpy as np
import math

def MLE(data, labels):
    """
    Please follow the format of fA_params and fA_pi, and return the
    result in the format of (fA_params,fA_pi)
    :type data: 2D numpy array
    :type labels: 1D numpy array
    :rtype: tuple
    """
    set_of_labels = set()
    for l in labels:
        set_of_labels.add(l)
    pi = np.asarray([sum(labels == l)/float(len(labels)) for l in set_of_labels])

    dim_data = data.shape[1]
    params = np.zeros((2,len(set_of_labels),dim_data),dtype = float)
    #mu:0; sigma:1
    for l in set_of_labels:
        ind = labels == l
        #label starts from 1
        params[0,l-1] = np.mean(data[ind,:], axis = 0)
        params[1,l-1] = np.var(data[ind,:], axis = 0)
    return (params, pi)

#for step1
MLE_result = MLE(training_data, training_label)

def gaussian(val, mean, var):
    e = math.exp(-(math.pow(val-mean,2)/(2*var)))
    return e / math.sqrt(2*math.pi*var)

def apply(data,params, pi):
    """
    :type data: 1D numpy array
    :type params: 3D numpy array for mu and sigma^2
    :type pi: 1D numpy array for pi
    :rtype: 1D numpy array, the normalized predicted distribution
    """
    prob = np.copy(pi)
    for i in range(len(pi)):
        for j in range(len(data)):
            prob[i] *= gaussian(data[j],params[0, i,j], params[1,i,j])
    return np.divide(prob, sum(prob))

params,pi = MLE_result
predicted_distr = np.asarray([apply(training_data[i], params, pi) for i in range(5)])

def MAP(data, labels, prior_params, pseudo_count):
    """
    :type data: 2D numpy array
    :type labels: 1D numpy array
    :type params: 3D numpy array for mu and sigma^2
    :type pseudo_count: 1D numpy array for pseudo_count, recall that this is fA_pi[1,:]
    :rtype:tuple, same format as MLE
    """
    set_of_labels = set()
    total_pseudo_count = sum(pseudo_count)
    for l in labels:
        set_of_labels.add(l)
    denominator = float(len(labels)+ total_pseudo_count - len(set_of_labels))
    pi = np.asarray([(sum(labels == l) + pseudo_count[l-1] - 1)/denominator for l in set_of_labels])

    dim_data = data.shape[1]
    params = np.zeros((2,len(set_of_labels),dim_data),dtype = float)
    #mu:0; sigma:1
    for l in set_of_labels:
        ind = labels == l
        #label starts from 1
        params[1,l-1] = np.var(data[ind,:], axis = 0)
        ratio = np.divide(params[1,l-1], prior_params[1,l-1])
        params[0,l-1] = np.divide(prior_params[0,l-1]*ratio + np.sum(data[ind,:], axis = 0), ratio + sum(ind))
    return (params, pi)

MAP_result = MAP(training_data, training_label, fA_params, fA_pi[1])

#params,pi = MAP_result
#predicted_distr = [apply(training_data[i], params, pi) for i in range(5)]

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
    num_per_fold = len(training_label)/k
    acc = np.zeros(k)
    for i in range(k):
        bool_indices = np.ones(len(training_label), dtype = bool)
        start_ind = int(i*num_per_fold)
        end_ind = int(min(num_per_fold*(i+1), len(training_label)))
        bool_indices[start_ind:end_ind] = False
        ith_training_data = training_data[bool_indices]
        ith_training_label = training_label[bool_indices]
        ith_testing_data = training_data[np.logical_not(bool_indices)]
        ith_testing_label = training_label[np.logical_not(bool_indices)]

        params, pi = MAP(ith_training_data, ith_training_label, prior_params, prior_pi)
        predicted_distr = np.asarray([apply(ith_testing_data[i], params, pi) for i in range(len(ith_testing_label))])
        predicted_label = np.argmax(predicted_distr, axis = 1)+1
        acc[i] = sum(predicted_label == ith_testing_label)/len(predicted_label)
    return np.average(acc)

fB_params, fB_pi = MLE_result
fB_count = fB_pi * len(training_label)
priors = [(fA_params, fA_pi[1]), (fB_params, fB_count), (fC_params, fC_pi[1])]
accuracies = np.asarray([CV(training_data, training_label, prior[0], prior[1], 5) for prior in priors])
best_prior = priors[np.argmax(accuracies)]
#best_prior = priors[1]

params, pi = MAP(training_data, training_label, best_prior[0], best_prior[1])
predicted_distr2 = np.asarray([apply(testing_data[i], params, pi) for i in range(len(testing_label))])
predictions = np.argmax(predicted_distr2, axis = 1)+1
acc = sum(predictions == testing_label)/len(predictions)
