import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
     # Scatter plot for class 1 (label 1)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', marker='o', c='blue')

    # Scatter plot for class -1 (label -1)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label='Class -1', marker='x', c='red')

    # Set labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2-D Scatter Plot of Training Features (Excluding Bias Term)')

    # Add legend
    plt.legend()

    # Save the plot to 'train_features.png'
    plt.savefig('train_features.png')

    # Show the plot (optional)
    plt.show()

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    # Scatter plot for data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, label='Data Points')

    # Plot decision boundary
    if W[2] != 0:
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_vals = (-W[0] - W[1] * x_vals) / W[2]
        plt.plot(x_vals, y_vals, '-r', label='Decision Boundary')

    plt.title('Sigmoid Model Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    # Save the plot
    plt.savefig('train_result_sigmoid.png')
    plt.show()

def visualize_result_multi(X, y, W):
    """This function is used to plot the softmax model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].
    
    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    """
    # Create a meshgrid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Generate predictions for each point on the grid
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W[1:3,:])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Softmax Model Visualization')
    plt.savefig('train_result_softmax.png')
    plt.show()
    

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

    #    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
	
    # Explore different hyper-parameters.
    ### YOUR CODE HERE

    results_dict = {}
	
    learning_rates = [0.1, 0.5, 1.0]
    max_iters = [50, 100, 200]
	
    # List of optimization methods
	
    optimization_methods = ['BGD', 'SGD', 'miniBGD']

    for method in optimization_methods:
        for lr in learning_rates:
            for max_iter in max_iters:
                # Create a new instance of logistic_regression for each combination of hyperparameters
                logisticR_classifier = logistic_regression(learning_rate=lr, max_iter=max_iter)

                # Train the model based on the optimization method
                if method == 'BGD':
                    logisticR_classifier.fit_BGD(train_X, train_y)
                elif method == 'SGD':
                    logisticR_classifier.fit_SGD(train_X, train_y)
                elif method == 'miniBGD':
                    logisticR_classifier.fit_miniBGD(train_X, train_y, batch_size=10)  # You can adjust batch_size

                # Store results
                results_dict[(method, lr, max_iter)] = logisticR_classifier.score(train_X, train_y)


    # Find the best hyperparameters
    best_hyperparams = max(results_dict, key=results_dict.get)
    best_method, best_learning_rate, best_max_iter = best_hyperparams
    print("best hyperparameters")
    print(best_hyperparams)
    

    ### END YOUR CODE


    ### YOUR CODE HERE

    # Visualize the 'best' model after training.
    best_logisticR = logistic_regression(learning_rate=best_learning_rate, max_iter=best_max_iter)

    if best_method == 'BGD':
        best_logisticR.fit_BGD(train_X, train_y)
    elif best_method == 'SGD':
        best_logisticR.fit_SGD(train_X, train_y)
    elif best_method == 'miniBGD':
        best_logisticR.fit_miniBGD(train_X, train_y, batch_size=10)
	
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    # ------------Data Preprocessing------------
	# Read data for testing.
    
    raw_test, label_test = load_data(os.path.join(data_dir, test_filename))

    # Preprocess test data
    test_X_all = prepare_X(raw_test)
    test_y_all, test_idx = prepare_y(label_test)

    # For binary case, only use data from '1' and '2'
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]

    # Set labels to 1 and -1
    test_y[np.where(test_y == 2)] = -1

    test_accuracy = best_logisticR.score(test_X, test_y)
    print("Test Accuracy with Best Model:", test_accuracy)

    ### END YOUR CODE
	
    
    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
	
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    learning_rates = [0.1, 0.5, 1.0]
    max_iters = [50, 100, 200,10000]

    best_accuracy = 0
    best_logisticR_multi = None

    for lr in learning_rates:
        for max_iter in max_iters:
            logisticR_classifier_multi = logistic_regression_multiclass(learning_rate=lr, max_iter=max_iter, k=3)
            logisticR_classifier_multi.fit_miniBGD(train_X, train_y, 10)
            accuracy = logisticR_classifier_multi.score(valid_X, valid_y)

            print(f"Learning Rate: {lr}, Max Iterations: {max_iter}, Accuracy: {accuracy}")

            # Keep track of the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_logisticR_multi = logisticR_classifier_multi
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, best_logisticR_multi.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    test_X = test_X_all
    test_y = test_y_all

    test_accuracy = best_logisticR_multi.score(test_X, test_y)
    print("Test Accuracy with Best Model for multi class:", test_accuracy)
    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
	
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    softmax_classifier = logistic_regression_multiclass(learning_rate=0.1, max_iter=10000, k=2)
    softmax_classifier.fit_miniBGD(train_X, train_y, batch_size=32)
    # Evaluate softmax classifier
    softmax_accuracy = softmax_classifier.score(valid_X, valid_y)
    print("Softmax Classifier Accuracy:", softmax_accuracy)

    ### END YOUR CODE
	
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 

    #####       set lables to -1 and 1 for sigmoid classifer
	
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    sigmoid_classifier = logistic_regression(learning_rate=0.1, max_iter=10000)
    sigmoid_classifier.fit_miniBGD(train_X, train_y, batch_size=32)

    # Evaluate sigmoid classifier
    sigmoid_accuracy = sigmoid_classifier.score(valid_X, valid_y)
    print("Sigmoid Classifier Accuracy:", sigmoid_accuracy)
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE

    # For softmax classifier
    batch_size = 10
    softmax_classifier = logistic_regression_multiclass(learning_rate=0.1, max_iter=10000, k=2)
    softmax_classifier.fit_miniBGD(train_X, train_y, batch_size=batch_size)

    #softmax_grandients = softmax_classifier._gradient(train_X[:batch_size], train_y[:batch_size])
    softmax_weights = softmax_classifier.get_params()

    # Set learning rate for sigmoid classifier
    learning_rate = 0.1

    # For sigmoid classifier
    sigmoid_classifier = logistic_regression(learning_rate=0.1, max_iter=10000)
    sigmoid_classifier.fit_miniBGD(train_X, train_y, batch_size=10)
    #sigmoid_gradients = sigmoid_classifier._gradient(train_X[:batch_size], train_y[:batch_size])
    sigmoid_weights = sigmoid_classifier.get_params()

    print("Softmax weights: ", softmax_weights)
    print("Sigmoid weights: ", sigmoid_weights)

    # To obtain w_1 - w_2 = w for all training steps, we need to set the learning rate for the sigmoid classifier to be equal to half of the learning rate for the softmax classifier
    learning_rate_sigmoid = 0.5 * learning_rate

    # Fit sigmoid classifier with the new learning rate
    sigmoid_classifier_updated = logistic_regression(learning_rate=learning_rate_sigmoid, max_iter=10000)
    sigmoid_classifier_updated.fit_miniBGD(train_X, train_y, batch_size=10)

    #sigmoid_gradients_updated = sigmoid_classifier_updated._gradient(train_X[:batch_size], train_y[:batch_size])
    sigmoid_weights_updated = sigmoid_classifier_updated.get_params()

    print("Sigmoid weights updated: ", sigmoid_weights_updated)

    ### END YOUR CODE

# ------------End------------
    

if __name__ == '__main__':
	main()
    
    
