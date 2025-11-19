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
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):
    print(f"shape of training images: {np.shape(trainingImages)}")
    print(f"shape of training labels: {np.shape(trainingLabels)}")
    print(f"shape of testing images: {np.shape(testingImages)}")
    print(f"shape of testing labels: {np.shape(testingLabels)}")

    Xtilde = np.column_stack([trainingImages, np.ones(len(trainingImages))])
    # print(np.shape(Xtilde))
    # print(Xtilde)

    num_classes = np.unique(trainingLabels).shape[0]

    Wtilde = np.random.normal(0, 1e-5, size=(Xtilde.shape[1], num_classes))
    # print(np.shape(W))
    # print(W)

    Ztilde = Xtilde @ Wtilde
    # print(np.shape(Ztilde))
    # print(Ztilde)

    Yhat = softmax(Ztilde)
    # print(np.shape(Yhat))
    # print(Yhat)

    Y = one_hot(trainingLabels, num_classes)

    num_batches = np.shape(Xtilde)[0] // batchSize
    E = 10
    for epoch in range(E):
        for i in range(num_batches):
            start = i * batchSize
            end = start + batchSize

            X_batch = Xtilde[start:end]
            Y_batch = Y[start:end]

            grad = grad_ce(Wtilde, X_batch, Y_batch, alpha)
            Wtilde -= epsilon * grad

    return Wtilde

def compute_accuracy(X, Y, W, print_flag=False):
    if print_flag:
        print(np.shape(X))
        print(np.shape(Y))
        print(np.shape(W))

    Z = X @ W
    Yhat = softmax(Z)
    return np.mean(np.argmax(Yhat, axis=1) == Y)

def f_ce(Y, Yhat):
    return -np.mean(np.sum(Y * np.log(Yhat), axis=1))

def grad_ce(W, X, Y, alpha=0.):
    n = X.shape[0]
    Z = X @ W
    Yhat = softmax(Z)
    grad = (1 / n) * X.T @ (Yhat - Y)

    if alpha > 0:
        reg_grad = np.zeros_like(W)
        reg_grad[:-1] = (alpha / n) * W[:-1]  # don't regularize bias
        grad += reg_grad

    return grad

def one_hot(y, num_classes: int):
    return np.eye(num_classes)[y]

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)  # stability
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy")
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy")
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...

    x = np.array([1, 2, 3])
    y = np.array([1, 4, 9])
    w = trainPolynomialRegressor(x, y, 3)
    yhat = 0
    a = 2
    for i in range(len(w)):
        yhat += w[i] * a ** i

    # print(yhat)

    # Change from 0-9 labels to "one-hot" binary vector labels. For instance, 
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    # ...

    # Train the model
    Wtilde = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, alpha=.1)[:-1]

    # print(np.shape(W))
    # print(np.shape(trainingImages))
    # print(np.shape(trainingLabels))

    print(compute_accuracy(trainingImages, trainingLabels, Wtilde))

    # Visualize the vectors
    # ...
