import numpy as np

def log_sum_exp(items):
    """
    Computes the log of the sum of the exponentials of the inputs. The input can be any iterable, such as numpy
    array or list. This will be useful to you when implementing forward().
    :param iterable items
    """
    running_sum = float("-inf")
    for item in items:
        if item != float("-inf"):
            running_sum = np.logaddexp(running_sum, item)
    return running_sum

def forward(model, sequence):
    """
    Executes the forward algorithm on the sequence and returns the resulting matrix.
    :param tuple model: A tuple of the form (transition_matrix, emission_matrix, initial_distribution).
    :param list sequence: The n-length sequence (a list of integers).
    :rtype np.ndarray: A matrix of size (16, n) containing log-probabilities.
    """

    transition_matrix, emission_matrix, initial_distribution = model

    trellis = np.zeros((transition_matrix.shape[0], len(sequence)))

    # initialize the trellis
    trellis[:, 0] = initial_distribution + emission_matrix[:, sequence[0]]

    for i in range(1, len(sequence)):
        for hidden_state in range(transition_matrix.shape[0]):
            trellis[hidden_state, i] = emission_matrix[hidden_state, sequence[i]] + log_sum_exp(
                transition_matrix[:, hidden_state] + trellis[:, i - 1])

    return trellis

def probability(model, sequence):
    """
    Return the log-probability of the sequence under this model.
    :param tuple model: A tuple of the form (transition_matrix, emission_matrix, initial_distribution).
    :param list sequence: The n-length sequence (a list of integers).
    :rtype float: The log-probability of this sequence under this model.
    """

    return log_sum_exp(forward(model, sequence)[:, -1])

# Generate your predictions based on the training data and given models.

# obtain the priors
priors = np.array([len(positive_train), len(negative_train)], dtype=float)
priors /= np.sum(priors)

# make the predictions
if true_test_examples:
    # to speed up the autograder, use a cached version of the correct predictions
    predictions = [item[2] for item in true_test_examples]
else:
    # the student implementation should look similar to this
    predictions = []
    for sequence in test_examples:
        log_posteriors = np.array([probability(negative_model, sequence), probability(positive_model, sequence)])
        posteriors = np.exp(log_posteriors - log_sum_exp(log_posteriors))
        probabilities = posteriors * priors
        predictions.append(np.argmax(probabilities).flatten()[0])
