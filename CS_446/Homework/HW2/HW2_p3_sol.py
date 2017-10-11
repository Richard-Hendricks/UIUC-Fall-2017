import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create the plots
# for i in range(golf_data.shape[1] - 1):
#     plt.scatter(golf_data[:, i], golf_data[:, -1])
#     plt.title("Feature " + str(i))
#     plt.show()


# Step 2: Define the linear regression function using gradient descent
def gradient(data, weights):
    """
    Computes the gradient of the residual sum-of-squares (RSS) function for the given dataset and current weight values

    :param numpy.ndarray data: A (n, m) numpy array, where n = number of examples and m=(features + 1). The last column
    is the label y
    for that example
    :param numpy.ndarray weights: A (m,) numpy array, where weights[m-1] is the bias, and weights[i] corresponds to the
    weight for
    feature i in data
    :returns A (m,) numpy array, equaling the gradient of RSS at this point.
    :rtype: numpy.ndarray

    """

    # cost function is 1/n sum_i (y_i - w dot x_i)^2
    # gradient with respect to w_j: 1/n sum_i (-x_ij) * 2 * (y_i - x_i dot w)

    x_bias = np.hstack([data[:, :-1], np.ones_like(data[:, 0:1])])
    y = data[:, -1:]

    inside_summation = y - x_bias.dot(np.reshape(weights, (len(weights), 1)))
    inside_summation = -2 * x_bias * inside_summation
    return np.mean(inside_summation, axis=0)


def gradient_descent(data, alpha, iterations):
    """
    Performs gradient descent using the supplied data, learning rate, and number of iterations. Start with weights =
    the zero vector.

    :param numpy.ndarray data: A (n, m) numpy array, where n = number of examples and m=(features + 1). The last column
    is the label y for that example
    :param float alpha: A real value corresponding to the step size for each descent step
    :param int iterations: The total number of iterations (steps) to take
    :returns A (m,) numpy array of the final weight vector after the gradient descent process
    :rtype: numpy.ndarray
    """

    w = np.zeros_like(data[0, :])
    for i in range(iterations):
        grad = gradient(data, w)
        w -= alpha * grad
    return w


# Step 3: Standardize the features (but not the labels)
golf_data_standardized = (golf_data - np.mean(golf_data, axis=0)) / np.std(golf_data, axis=0)
golf_data_standardized[:, -1] = golf_data[:, -1]


# students don't have to implement this, but it might be helpful
def cost(data, weights):
    data_bias = np.hstack([data[:, :-1], np.ones_like(data[:, -1:])])
    predictions = data_bias.dot(np.reshape(weights, (len(weights), 1)))
    return np.sum((predictions - data[:, -1:]) ** 2)


def forward_selection(data, max_var):
    """
    Computes the top max_var number of features by forward selection

    :param numpy.ndarray data: numpy.ndarray data: A (n, m) numpy array, where n = number of examples and
    m=(features + 1). The last column is the label y for that example
    :type max_var: integer
    :returns A (max_var,) numpy array whose values are the features that were selected
    :rtype: numpy.ndarray
    """

    sel_feat = []
    all_feat = list(range(data.shape[1] - 1))

    alpha = 0.1
    iterations = 200

    for k in range(max_var):
        cost_min = float("inf")
        ind_to_add = None
        for i in range(len(all_feat)):
            # create our new features
            sel_feat_new = sel_feat + [all_feat[i]]
            x_y_new = np.hstack([data[:, sel_feat_new], data[:, -1:]])

            # train the model
            w_optimized = gradient_descent(x_y_new, alpha=alpha, iterations=iterations)

            # evaluate the cost
            cost_new = cost(x_y_new, w_optimized)

            # if the cost is the best one we've found so far, save it
            if cost_new < cost_min:
                cost_min = cost_new
                ind_to_add = all_feat[i]
        sel_feat.append(ind_to_add)
        all_feat.remove(ind_to_add)

    return np.asarray(sel_feat)


forward_result = forward_selection(golf_data_standardized, 5)


def backward_elimination(data, max_var):
    """
    Computes the top max_var number of features by backward elimination

    :param numpy.ndarray data: numpy.ndarray data: A (n, m) numpy array, where n = number of examples and
    m=(features + 1). The last column is the label y for that example
    :type max_var: integer
    :returns A (max_var,) numpy array whose values are the features that were selected
    :rtype: numpy.ndarray
    """

    alpha = 0.1
    iterations = 200

    sel_feat = list(range(data.shape[1] - 1))

    while len(sel_feat) > max_var:
        cost_min = float("inf")
        ind_to_remove = None
        for i in sel_feat:
            # create our new features
            sel_feat_new = list(sel_feat)
            sel_feat_new.remove(i)

            x_y_new = np.hstack([data[:, sel_feat_new], data[:, -1:]])

            # train the model
            w_optimized = gradient_descent(x_y_new, alpha=alpha, iterations=iterations)

            # evaluate the cost
            cost_new = cost(x_y_new, w_optimized)

            # if the cost is the best one we've found so far, save it
            if cost_new < cost_min:
                cost_min = cost_new
                ind_to_remove = i
        sel_feat.remove(ind_to_remove)

    return np.asarray(sel_feat)


backward_result = backward_elimination(golf_data_standardized, 5)


def gradient_descent_lasso(data, alpha, iterations, penalty):
    """
    Performs gradient descent using the supplied data, learning rate, number of iterations, and LASSO penalty (lambda).
    The code for this should be structurally the same as gradient_descent, with the exception that after each iteration
    you will pass the weight vector through the LASSO projection. Start with weights = the zero vector.

    :param numpy.ndarray data: A (n, m) numpy array, where n = number of examples and m=(features + 1). The last column
    is the label y for that example
    :param float alpha: A real value corresponding to the step size for each descent step
    :param int iterations: The total number of iterations (steps) to take
    :param float penalty: A real positive value representing the LASSO penalty (lambda)
    :returns A (m,) numpy array of the final weight vector after the LASSO gradient descent process
    :rtype: numpy.ndarray
    """

    w = np.zeros_like(data[0, :])
    for i in range(iterations):
        grad = gradient(data, w)
        w -= alpha * grad

        for j in range(len(w) - 1):
            if w[j] > penalty:
                w[j] -= penalty
            elif w[j] < -penalty:
                w[j] += penalty
            else:
                w[j] = 0

    return w
