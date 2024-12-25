import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.W = None

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """


        n_samples, n_features = X.shape

		### YOUR CODE HERE

        self.assign_weights(np.zeros(n_features))
        for _ in range(self.max_iter):
            gradient = np.zeros(n_features)
            for i in range(len(X)):
                gradient += self._gradient(X[i], y[i])

            self.W -= self.learning_rate * gradient / len(X)
		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.assign_weights(np.zeros(n_features))

        for _ in range(self.max_iter):
            indices = np.random.choice(len(X), batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            gradient = np.zeros(n_features)
            for i in range(len(X_batch)):
                gradient += self._gradient(X_batch[i], y_batch[i])

            self.W -= self.learning_rate * gradient / batch_size
		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.assign_weights(np.zeros(n_features))

        for _ in range(self.max_iter):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(len(X_shuffled)):
                self.W -= self.learning_rate * self._gradient(X_shuffled[i], y_shuffled[i])
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        #exp_term = np.exp(-_y * np.dot(self.W, _x))
        #return -_y * _x / (1 + exp_term)
        #v1
        exp_term = np.exp(-_y * np.dot(self.W, _x))
        sigmoid = 1 / (1 + exp_term)

        # Gradient clipping
        gradient = -_y * _x * (1 - sigmoid)
        gradient = np.clip(gradient, -1e3, 1e3) # To avoid overflow in e
        return gradient
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        exp_term = np.exp(-np.dot(X, self.W))
        preds_proba = np.vstack([1 / (1 + exp_term), exp_term / (1 + exp_term)]).T
        return preds_proba
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        preds = np.sign(np.dot(X, self.W))
        return preds
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return accuracy
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

