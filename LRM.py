#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        #k = np.max(labels) + 1  # Number of classes
        k = self.k

        # Convert labels to integers and then to one-hot encoding
        labels_one_hot = np.zeros((len(labels), k), dtype=int)
        labels_one_hot[np.arange(len(labels)), labels.astype(int)] = 1

        # Initialize weights
        self.W = np.zeros((n_features, k))

        for epoch in range(self.max_iter):
            # Shuffle the data
            shuffle_indices = np.random.permutation(n_samples)
            X_shuffled = X[shuffle_indices]
            labels_shuffled = labels_one_hot[shuffle_indices]

            for i in range(0, n_samples, batch_size):
                # Mini-batch
                X_batch = X_shuffled[i:i+batch_size]
                labels_batch = labels_shuffled[i:i+batch_size]

                # Compute gradient and update weights
                gradient = self._gradient(X_batch, labels_batch)
                self.W -= self.learning_rate * gradient

        return self
		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        predictions = self.softmax(np.dot(_x, self.W))
        _g = np.dot(_x.T, (predictions - _y)) / len(_x)
        return _g
		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        predictions = np.argmax(self.softmax(np.dot(X, self.W)), axis=1)
        return predictions
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        predictions = self.predict(X)
        accuracy = np.mean(predictions == labels)
        return accuracy
		### END YOUR CODE

