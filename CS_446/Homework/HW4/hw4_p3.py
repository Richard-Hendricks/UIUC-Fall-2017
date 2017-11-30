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

    # Implement me!
    pass

def probability(model, sequence):
    """
    Return the log-probability of the sequence under this model.
    :param tuple model: A tuple of the form (transition_matrix, emission_matrix, initial_distribution).
    :param list sequence: The n-length sequence (a list of integers).
    :rtype float: The log-probability of this sequence under this model.
    """

    transition_matrix, emission_matrix, initial_distribution = model

    # Implement me!
    pass


# Generate your predictions based on the training data and given models.
predictions = [0] * len(test_examples)  # Implement me!
