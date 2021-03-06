#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:57:42 2017

@author: Yiming
"""

import numpy as np
import math

def split_by_feature(objects, feature_index):
    '''
    Groups the given objects by feature 'feature_index'

    :param list objects: The list of objects to be split
    :param int feature_index: The index of the feature by which to split
    :return: A dictionary of (feature value) => (list of objects with that feature value)
    :rtype: dict
    '''

    # find all the distinct values of this feature
    distinct_values = set([object[0][feature_index] for object in objects])

    # create a mapping of value => list of objects with that value
    mapping = {
        value: [object for object in objects if object[0][feature_index] == value]
        for value in distinct_values
    }
    return mapping
# totally 6 features, feature_index can be 0, 1, 2, 3, 4, 5
# if 0, print out a dict that contains all rows with that value in that column 0
feature0 = split_by_feature(train_objects, 0)

def group(objects):
    '''
    Groups the given objects by their labels, returning the weighted count of each label (i.e. it returns the sum
    of the weights of all objects with this label, not the count of the objects)

    :param list objects: The list of objects to be grouped
    :return: A dictionary of (label value) => (sum of object weights with this label)
    :rtype: dict
    '''

    return {
        label: sum([object[2] for object in objects if object[1] == label])
        for label in set([object[1] for object in objects])
    }

pre_split = group(train_objects) # {"b'acc'": 3, "b'unacc'": 7}


def split_quality(before_split, split_results, evaluation_function):
    '''
    Takes the before and after of a split, along with an evaluation function, and returns the quality of this
    split (higher values mean better splits)

    :param list before_split: The full list of objects before the split
    :param dict split_results: The dictionary of (feature_value) => (list of objects with that feature value)
    that was returned from split_by_feature
    :return: The result of the evaluation function for this split
    :rtype: float
    '''

    # group these both by the label => count
    return evaluation_function(group(before_split),
                               [group(subset) for feature_value, subset in split_results.items()])



def dominant_label(objects):
    '''
    Accepts a list of objects and returns the most common label. It takes into account the object weights. Ties
    are broken in an undefined manner.

    :param list objects: The list of objects
    :return: The label that appeared most frequently in these objects
    :rtype: object
    '''

    grouping = group(objects)
    return (list(sorted(grouping.items(), key=lambda group: -group[1])) or [None])[0][0]

dominant_label(train_objects) # "b'unacc'"

def train_tree(objects, evaluation_function, max_depth=None):
    '''
    Trains a decision tree with a specific split evaluation function and a maximum depth

    :param list objects: The list of training objects
    :param callable evaluation_function: The function to be used to evaluate the quality of the splits (either
    gini_gain or information_gain)
    :param int max_depth: The maximum depth (number of tests) to be applied to the data
    
    :return: A tree object. In cases of a homogeneous set of data or when max_depth=0, this will return an object
    representing the label in this data set. In other cases, it will return a dictionary of {feature => (the
    feature index to split on), actions => (a dictionary of possible feature values => the subsequent tree for
    that value)}.
    :rtype: object
    '''

    # if there are no objects to split, we can't make a guess
    if len(objects) == 0:
        return None

    # if we've hit the max_depth limit, just return the majority label at this point
    if max_depth == 0:
        return dominant_label(objects)

    # find the next split by looping through all features. we will assume that 'objects' is a
    # square table, so the feature indices can be taken from the first example
    best_quality = 0
    best_feature = None
    best_split = None
    for feature_index in range(len(objects[0][0])):
        split_results = split_by_feature(objects, feature_index)

        # evaluate our dataset after the split
        quality = split_quality(objects, split_results, evaluation_function)

        if quality > best_quality:
            best_quality = quality
            best_feature = feature_index
            best_split = split_results

    # perform whichever split we determined was best
    if best_feature is None:
        # no split made any improvement. return the dominant label in this set
        return dominant_label(objects)
    else:
        # a split was made. drill down further
        return {
            "feature": best_feature,
            "actions": {
                feature_value: train_tree(objects, evaluation_function, None if max_depth is None else max_depth - 1)
                for feature_value, objects in best_split.items()
            }
        }


def evaluate_single(tree, object):
    '''
    Obtains a prediction for a given object using the given decision tree. Note that although the object's true
    label is technically passed through 'object', it is never used.

    :param object tree: The tree being used to evaluate the object
    :param tuple object: The object to be evaluated. The ground truth label, object[1], will always be set to None.
    :return: The label corresponding to the prediction for this class, or None if the decision tree did not output a label
    :rtype: object
    '''

    if tree is None or not isinstance(tree, dict):
        # we've reached a leaf!
        return tree
    elif object[0][tree["feature"]] not in tree["actions"]:
        # we didn't see this feature value when training the tree, so just return None for now
        return None
    else:
        # recurse!
        return evaluate_single(tree["actions"][object[0][tree["feature"]]], object)


# have them implement this
def gini_gain(pre_split, post_split):
    '''
    Evaluates the quality of a split using the Gini Impurity metric

    :param dict pre_split: A dictionary of (label) => (weighted count) corresponding to the number of instances
    of this label before the split
    :param list post_split: A list of dictionaries following the same format as pre_split. Each entry in this list
    corresponds to the new distribution after the split.
    :return: A real (non-negative) number, where a higher value indicates a higher purity post-split compared to
    pre-split
    :rtype: float
    '''
    # counts for each label
    n_samples = sum(pre_split.values())
    post_counts = []
    for i in range(len(post_split)):
        post_counts.append(sum(post_split[i].values()))
    
    # pre_GINI
    pre_gini = 1 - sum([pow(v / n_samples,2) for k, v in pre_split.items()])
    
    # post GINI
    post_gini_l = []
    for i in range(len(post_split)):
        G = 0
        for j in post_split[i].values():
            G = G + (j / post_counts[i]) ** 2
        G = 1 - G
        post_gini_l.append(G)
    
    post_gini = 0
    for i in range(len(post_split)):
        post_gini = post_gini + post_counts[i] / n_samples * post_gini_l[i]
    
    return (pre_gini - post_gini)
    pass

# have them implement this
def information_gain(pre_split, post_split):
    '''
    Evaluates the quality of a split using the Information Gain metric

    :param dict pre_split: A dictionary of (label) => (weighted count) corresponding to the number of instances
    of this label before the split
    :param list post_split: A list of dictionaries following the same format as pre_split. Each entry in this list
    corresponds to the new distribution after the split.
    :return: A real (non-negative) number, where a higher value indicates a higher purity post-split compared to
    pre-split
    :rtype: float
    '''
       # counts for each label
    n_samples = sum(pre_split.values())
    post_counts = []
    for i in range(len(post_split)):
        post_counts.append(sum(post_split[i].values()))
    
    # pre_INFO
    pre_info = - sum([(v / n_samples) * np.log2(v / n_samples) for k, v in pre_split.items()])
    
    # post GINI
    post_info_l = []
    for i in range(len(post_split)):
        H = 0
        for j in post_split[i].values():
            H = H + (j / post_counts[i]) * np.log2(j / post_counts[i])
        H = -H
        post_info_l.append(H)
    
    post_info = 0
    for i in range(len(post_split)):
        post_info = post_info + post_counts[i] / n_samples * post_info_l[i]
    
    return (pre_info - post_info)
    pass

def evaluate_tree(tree, objects):
    '''
    Evaluates the weighted accuracy of a decision tree on a list of objects. When calling 'evaluate_single', the
    true label will not be passed.

    :param object tree: The tree being used to evaluate the objects
    :param list objects: The objects to be used as the test set
    :return: A real number between 0 and 1, where 1 corresponds to a perfect ability to predict
    :rtype: float
    '''

    errors = 0
    total = 0
    for object in objects:
        total += object[2]
        if evaluate_single(tree, (object[0], None) + object[2:]) != object[1]:
            errors += object[2]

    return 1 - errors / total


def adaboost(objects, iterations, stump_depth):
    '''
    Trains a set of decision stumps using AdaBoost

    :param list objects: The training data
    :param int iterations: How many decision stumps we should train
    :param int stump_depth: The depth of each tree trained
    :return: A list of tuples (tree, weight), which is the model learned
    :rtype: list
    '''
    # See SAMME's paper as reference
    # train_tree(train_objects, gini_gain, 2)
    # Initialize weights
    w = list(np.ones(len(objects)) * (1 / len(objects)))
    result = []
    
    for m in range(iterations):
        error = []
        tree = train_tree(objects, gini_gain, stump_depth)
        
        for i in range(len(objects)):
            if evaluate_single(tree, (objects[i][0], None) + objects[i][2:]) != objects[i][1]:
                error.append(w[i] * 1)
            else:
                error.append(w[i] * 0)
                
        error = sum(error) / sum(w)
        alpha = np.log((1-error)/error) + np.log(4 - 1)
        
        for i in range(len(objects)):
            if evaluate_single(tree, (objects[i][0], None) + objects[i][2:]) != objects[i][1]:
                w[i] = w[i] * np.exp(alpha * 1)
            else:
                w[i] = w[i] * np.exp(alpha * 0)
    
        # Re-normalize w
        w = [x / sum(w) for x in w]
        
        # Update weights in objects
        for i in range(len(objects)):
            objects[i] = (objects[i][0], objects[i][1], w[i])
        
        result.append((tree, alpha))
        
    return result
    # Implement me!
    pass


def evaluate_adaboost_single(trees, object):
    '''
    Takes the learned AdaBoost model and an object and computes its predicted class label

    :param list trees: The AdaBoost model returned from adaboost()
    :param tuple object: The test example for which to obtain a prediction. The ground truth label, object[1],
    will always be set to None.
    :return: The predicted class label for 'object'
    :rtype: object
    '''
    pred = []
    alpha = []
    
    # Get predictions and weights from each tree
    for m in range(len(trees)):
        pred.append(evaluate_single(trees[m][0], object))
        alpha.append(trees[m][1])
    
    result = []
    for label in set(pred):
        temp = []
        
        for m in range(len(trees)):
            if pred[m] == label:
                temp.append(alpha[m])
        
        # Get a list containing each class and corresponding sum of weights
        result.append((label, sum(temp)))
        
    # return the class with largest weight
    return [k[0] for k in result if k[1] == max([x[1] for x in result])][0]
    pass

def evaluate_adaboost(trees, objects):
    '''
    Evaluates the weighted accuracy of a AdaBoost model on a list of objects. When calling 'evaluate_adaboost_single', the
    true label will not be passed.

    :param list trees: The AdaBoost model returned from adaboost()
    :param list objects: The objects to be used as the test set
    :return: A real number between 0 and 1, where 1 corresponds to a perfect ability to predict
    :rtype: float
    '''

    errors = 0
    total = 0
    for object in objects:
        total += object[2]
        if evaluate_adaboost_single(trees, (object[0], None) + object[2:]) != object[1]:
            errors += object[2]

    return 1 - errors / total
