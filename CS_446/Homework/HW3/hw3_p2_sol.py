import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, gamma=100):
    """
    RBF Kernel

    :type x1: 1D numpy array of features
    :type x2: 1D numpy array of features
    :type gamma: float bandwidth
    :rtype: float
    """
    # Student implementation here
    diff = x1 - x2
    return np.exp(-gamma*np.sum(np.dot(diff, diff)))

def w_dot_x(x, x_list, y_list, alpha, kernel_func=rbf_kernel):
    """
    Calculates wx using the training data

    :type x: 1D numpy array of features
    :type x_list: 2D numpy array of training features
    :type y_list: 1D numpy array of training labels where len(x) == len(y)
    :type alpha: 1D numpy array of int counts
    :type kernel_func: function that takes in 2 vectors and returns a float
    :rtype: float representing wx
    """
    # Student implementation here
    norm_const = max(1, np.sum(alpha))
    return np.sum(np.fromiter((y_list[i] * alpha[i] * kernel_func(xi, x) for (i, xi) in enumerate(x_list) if alpha[i] > 0), dtype=float))/(norm_const)

def predict(x, x_list, y_list, alpha):
    """
    Predicts the label {-1,1} for a given x

    :type x_list: 2D numpy array of training features
    :type y_list: 1D numpy array of training labels where len(x) == len(y)
    :rtype: Integer in {-1,1}
    """
    wx = w_dot_x(x, x_list, y_list, alpha)
    return np.sign(wx)

def pegasos_train(x_list, y_list, tol=0.01):
    """
    Trains a predictor using the Kernel Pegasos Algorithm

    :type x_list: 2D numpy array of training features
    :type y_list: 1D numpy array of training labels where len(x) == len(y)
    :rtype: 1D numpy array of int counts
    """
    # Alpha counts the number of times the traning example has been selected
    alpha = np.zeros(len(x_list), dtype=int)
    for t in range(num_samples_to_train):
        # Student implementation here
        rand_i = np.random.randint(len(x_list))
        dist = y_list[rand_i]*(w_dot_x(x_list[rand_i], x_list, y_list, alpha))
        if dist < tol:
            alpha[rand_i] += 1
    return alpha

def accuracy(x_list, y_list, alpha, x_test, y_test):
    """
    Calculates the proportion of correctly predicted results to the total

    :type x_list: 2D numpy array of training features
    :type y_list: 1D numpy array of training labels where len(x) == len(y)
    :type alpha: 1D numpy array of int counts
    :type x_test: 2D numpy array of test features
    :type y_test: 1D numpy array of test labels where len(x) == len(y)
    :rtype: float as a proportion of correct labels to the total
    """
    prediction = np.fromiter((predict(xi, x_list, y_list, alpha) for xi in x_test), x_test.dtype)
    return float(np.sum(np.equal(prediction, y_test)))/len(y_test)

# Student implementation here
final_alpha = pegasos_train(x_train, y_train)
print(np.sum(final_alpha))
print(accuracy(x_train, y_train, final_alpha, x_test, y_test))

# Plot Countour
res = 8
xplot = np.linspace(min(x_train[:, 0]), max(x_train[:, 0]), res)
yplot = np.linspace(min(x_train[:, 1]), max(x_train[:, 1]), res)
xx,yy = np.meshgrid(xplot, yplot)
xy = np.array([[x,y] for x in xplot for y in yplot])
prediction = np.fromiter((predict(xi, x_train, y_train, final_alpha) for xi in xy), x_test.dtype).reshape((res,res)).T
plt.contourf(xx,yy,prediction)


# Plot data
pos_x = np.array([x for i, x in enumerate(x_train) if y_train[i] == 1])
neg_x = np.array([x for i, x in enumerate(x_train) if y_train[i] == -1])
plt.scatter(pos_x[:, 0], pos_x[:, 1], label="+1")
plt.scatter(neg_x[:, 0], neg_x[:, 1], label="-1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()
