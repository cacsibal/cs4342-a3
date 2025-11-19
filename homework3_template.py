import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# PROBLEM 1
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    n = np.shape(x)[0]
    # print(n)

    X = np.zeros((n, d + 1))
    for i in range(d + 1):
        X[:, i] = x ** i

    return np.linalg.solve(X.T @ X, X.T @ y)

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):
    pass

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...

    x = np.array([1, 2, 3])
    y = np.array([1, 4, 9])
    w = trainPolynomialRegressor(x, y, 3)
    yhat = 2
    yhat = 0
    for i in range(len(w)):
        yhat += w[i] * a ** i

    print(yhat)

    # Change from 0-9 labels to "one-hot" binary vector labels. For instance, 
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    # ...

    # Train the model
    Wtilde = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, alpha=.1)
    
    # Visualize the vectors
    # ...
