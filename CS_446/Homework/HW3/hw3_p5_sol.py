import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    sigmoid function
    :param ndarray z
    """
    return 1.0/(1.0+np.exp(-z))

def predict(x, w, b):
    """
    Forward prediction of neural network
    :param ndarray x: num_feature x 1 numpy array
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :rtype int: label index, starting from 1
    """
    a = x
    for i in range(len(w)):
        l = np.dot(w[i],a) + b[i]
        a = sigmoid(l)
    return np.argmax(a) + 1

def accuracy(testing_data, testing_label, w, b):
    """
    Return the accuracy(0 to 1) of the model w, b on testing data
    :param ndarray testing_data: num_data x num_feature numpy array
    :param ndarray testing_label: num_data x 1 numpy array
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :rtype float: accuracy(0 to 1)
    """
    num_feature = len(testing_data[0])#.shape[1]
    predicts = np.asarray([predict(x.reshape((num_feature,1)),w, b) for x in testing_data], dtype = int)
    acc = sum(predicts == testing_label)/float(len(predicts))
    return acc


def gradient(x, y, w, b):
    """
    Compute gradient using backpropagation
    :param ndarray x: num_feature x 1 numpy array
    :param ndarray y: num_label x 1 numpy array
    :rtype tuple: A tuple contains the delta/gradient of weights and bias (dw, db)
                dw and db should have same format as w and b correspondingly
    """
    a_s= [x]
    db = [0,0]
    dw = [0,0]
    for i in range(len(w)):
        l = np.dot(w[i],a_s[i]) + b[i]
        a_s.append(sigmoid(l))
    dE_dak = a_s[2]-y
    dak_dlk = a_s[2] *(1-a_s[2])
    db[1] = delta = dE_dak * dak_dlk
    dw[1] = np.dot(db[1], a_s[1].T)

    dah_dlh = a_s[1] *(1-a_s[1])
    db[0] = delta = np.dot(w[1].T, delta) * dah_dlh
    dw[0] = np.dot(db[0], a_s[0].T)
    return (dw, db)

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
    num_feature = len(training_data[0])
    dw = [np.zeros(w_.shape) for w_ in w]
    db = [np.zeros(b_.shape) for b_ in b]
    for i in range(len(training_label)):
        x = np.reshape(training_data[i],(num_feature,1))
        y = np.zeros((num_label,1))
        y[training_label[i]-1] = 1.0
        tdw,tdb = gradient(x,y,w,b)
        for i in range(len(dw)):
            dw[i] += tdw[i]
            db[i] += tdb[i]
    for i in range(len(w)):
        w[i] -= eta * dw[i]/len(training_label)
        b[i] -= eta * db[i]/len(training_label)
    return (w,b)

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
    epoches_training_acc = []
    epoches_testing_acc = []
    for i in range(num_epochs):
        test_acc = accuracy(testing_data, testing_label, w, b)
        epoches_testing_acc.append(test_acc)
        train_acc = accuracy(training_data, training_label, w, b)
        epoches_training_acc.append(train_acc)

        w,b =single_epoch(w, b, training_data, training_label, eta, num_label)
    return (w,b, epoches_training_acc, epoches_testing_acc)

#BEGIN don't touch
num_label = 3
num_feature = len(training_data[0])
num_hidden_nodes = 50 #50 is not the best parameter, but we fix it here
step_sizes = [0.3,3,10]
#REFER the dimension and format here
#sizes =[num_feature, num_hidden_nodes, num_label]
#init_weights = [prng.randn(sizes[i+1], sizes[i]) for i in range(len(sizes)-1)]
#init_bias = [prng.randn(sizes[i+1],1) for i in range(len(sizes)-1)]
#END don't touch

#ATTENTION:
#If you are going to call batch_gradient_descent multiple times
#DO MAKE A DEEP COPY OF init_weights AND init_bias BEFORE CALLING!
#Or MAKE A DEEP COPY when use them in batch_gradient_descent

epoches = np.arange(0.,200.,1.)
for i,eta in enumerate(step_sizes):
    t_w = [np.copy(w) for w in init_weights_secret_copy]
    t_b = [np.copy(b) for b in init_bias_secret_copy]
    _, _, epoches_tr_accs, epoches_te_accs = batch_gradient_descent(t_w, t_b, training_data, training_label, eta, 3)
    plt.figure(i+1)
    tr_line, = plt.plot(epoches, epoches_tr_accs, 'r', label="training accuracy")
    te_line, = plt.plot(epoches, epoches_te_accs, 'g', label="testing accuracy")
    plt.legend(handles=[tr_line, te_line])
    plt.ylabel('Accuracies')
    plt.xlabel('Epoch Number')
plt.show()

weights = [np.copy(w) for w in init_weights_secret_copy]
bias = [np.copy(b) for b in init_bias_secret_copy]
weights, bias,_,_ = batch_gradient_descent(weights, bias, training_data, training_label, 3, 3)

#print('In the given cases, step_size = 3 is the best parameter')

#print('In the given cases step_size = 0.3 is too small for 200 epoches, and as a result, the model is not fully trained. Some students states that eta = 0.3 and the convergence is stuck at a local minimum, we will also accept this answer since in some graphs the accruacy does not change much at the first beginning 1000 epoches. However, if we extend epoch = 2000, you will find this model will finally achieve very good testing and training accuracy as well.')

#print('In the given cases, step_size = 10 is too large, which usually repeatly causes overshoot the minimum, and eventually diverges.')
