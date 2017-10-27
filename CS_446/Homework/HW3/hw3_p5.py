import numpy as np
import matplotlib.pyplot as plt

# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
def sigmoid(z):
    """
    sigmoid function
    :param ndarray z
    """
    return 1.0/(1.0+np.exp(-z))

# flatten list: [x for sublist in init_bias[0] for x in sublist]
z1 = init_weights[0].dot(x) + np.array([x for sublist in init_bias[0] for x in sublist])
a1 = sigmoid(z1)
z2 = init_weights[1].dot(a1) + np.array([x for sublist in init_bias[1] for x in sublist])
a2 = sigmoid(z2)
np.argmax(a2) + 1

def predict(x, w, b):
    """
    Forward prediction of neural network
    :param ndarray x: num_feature x 1 numpy array
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :rtype int: label index, starting from 1
    """
    #IMPLEMENT ME
    z1 = np.dot(w[0], x) + b[0]
    a1 = sigmoid(z1)
    z2 = np.dot(w[1], a1) + b[1]
    a2 = sigmoid(z2)
    
    return np.argmax(a2) + 1
    pass

def mypredict(x, w, b):
    """
    Forward prediction of neural network
    :param ndarray x: num_feature x 1 numpy array
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :rtype int: label index, starting from 1
    """
    #IMPLEMENT ME
    z1 = np.expand_dims(np.dot(w[0], x), axis = 1) + b[0]
    a1 = sigmoid(z1)
    z2 = np.dot(w[1], a1) + b[1]
    # z2 = np.dot(w[1], a1) + [x for sublist in b[1] for x in sublist]
    a2 = sigmoid(z2)
    
    return np.argmax(a2) + 1

def accuracy(testing_data, testing_label, w, b):
    """
    Return the accuracy(0 to 1) of the model w, b on testing data
    :param ndarray testing_data: num_data x 13 numpy array
    :param ndarray testing_label: num_data x 1 numpy array
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :rtype float: accuracy(0 to 1)
    """
    #IMPLEMENT ME
    result = []
    for i in range(len(testing_label)):
        result.append(mypredict(testing_data[i], w, b) == testing_label[i])

    return result.count(True) / len(result)
    pass


def gradient(x, y, w, b):
    """
    Compute gradient using backpropagation
    :param ndarray x: num_feature x 1 numpy array
    :param ndarray y: num_label x 1 numpy array
    :rtype tuple: A tuple contains the delta/gradient of weights and bias (dw, db)
                dw and db should have same format as w and b correspondingly
    """
    #IMPLEMENT ME
    pass


def single_epoch(w, b, training_data, training_label, eta, num_label):
    """
    Compute one epoch of batch gradient descent
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :param ndarray training_data: num_data x num_feature numpy array
    :param ndarray training_label: num_data x 1 numpy array
    :param float eta: step size
    :param int num_label: number of labels
    :rtype tuple: A tuple contains the updated weights and bias (w, b)
                w and b should have same format as they are pased in
    """
    #IMPLEMENT ME
    pass


def batch_gradient_descent(w, b, training_data, training_label, eta, num_label, num_epochs = 200):
    """
    Train the NN model using batch gradient descent
    You may NEED to modify this function for plotting
    You are FREE to add additional return results
    We WONT test this function in autograder
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :param ndarray training_data: num_data x num_feature numpy array
    :param ndarray training_label: num_data x 1 numpy array
    :param float eta: step size
    :param int num_label: number of labels
    :rtype tuple: A tuple contains the updated weights and bias (w, b)
                w and b should have same format as they are pased in
    """
    batch_size = 10
    for i in range(num_epochs):
        #if i%10 = 0:
            #test_acc = accuracy(testing_data, testing_label, w, b)
            #train_acc = accuracy(training_data, training_label, w, b))
            #print 'start epoch {}, train acc {} test acc {}'.format(i, train_acc, test_acc)
        w,b =single_epoch(w, b, training_data, training_label, eta, num_label)
    return (w,b)

#BEGIN don't touch
num_label = 3
num_feature = 13
num_hidden_nodes = 50 #50 is not the best parameter, but we fix it here
step_sizes = [0.3,3,10]
#REFER the dimension and format here
sizes =[num_feature, num_hidden_nodes, num_label]
init_weights = [np.random.rand(sizes[i+1], sizes[i]) for i in range(len(sizes)-1)]
init_bias = [np.random.rand(sizes[i+1],1) for i in range(len(sizes)-1)]
#END don't touch

#ATTENTION:
#If you are going to call batch_gradient_descent multiple times
#DO MAKE A DEEP COPY OF init_weights AND init_bias BEFORE CALLING!
#Or MAKE A DEEP COPY when use them in batch_gradient_descent

#weights, bias = batch_gradient_descent(init_weights, init_bias, training_data, training_label, step_sizes[-1], 3)
#print accuracy(training_data, training_label, weights, bias)
#print accuracy(testing_data, testing_label, weights, bias)
