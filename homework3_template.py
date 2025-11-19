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
    # print(f"shape of training images: {np.shape(trainingImages)}")
    # print(f"shape of training labels: {np.shape(trainingLabels)}")
    # print(f"shape of testing images: {np.shape(testingImages)}")
    # print(f"shape of testing labels: {np.shape(testingLabels)}")

    num_classes = np.unique(trainingLabels).shape[0]
    # print(f"number of unique classes: {num_classes}")

    Xtilde = np.column_stack([trainingImages, np.ones(trainingImages.shape[0])])
    # print(f"shape of Xtilde: {np.shape(Xtilde)}")

    Wtilde = np.random.normal(0, 1e-5, size=(Xtilde.shape[1], num_classes))
    # print(f"shape of Wtilde: {np.shape(Wtilde)}")

    shuffled_indices = np.arange(np.shape(Xtilde)[0])
    np.random.shuffle(shuffled_indices)

    Xtilde_shuffled = Xtilde[shuffled_indices]
    trainingLabels_shuffled = trainingLabels[shuffled_indices]

    Y_onehot = one_hot(trainingLabels_shuffled, num_classes)

    num_batches = np.shape(Xtilde)[0] // batchSize
    E = 30

    for e in range(E):
        for i in range(num_batches):
            # Get batch
            start_idx = i * batchSize
            end_idx = (i + 1) * batchSize

            X_batch = Xtilde_shuffled[start_idx:end_idx]
            Y_batch = Y_onehot[start_idx:end_idx]

            grad = gradfSoftmax(Wtilde, X_batch, Y_batch, alpha)
            Wtilde = Wtilde - epsilon * grad

    return Wtilde

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def gradfSoftmax(Wtilde, X, Y, alpha=0.):
    n = X.shape[0]

    Z = X @ Wtilde
    Yhat = softmax(Z)

    grad = (1 / n) * X.T @ (Yhat - Y)

    if alpha > 0:
        reg_grad = np.zeros_like(Wtilde)
        reg_grad[:-1] = (alpha / n) * Wtilde[:-1]
        grad += reg_grad

    return grad

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
    yhat = 0
    a = 2
    for i in range(len(w)):
        yhat += w[i] * a ** i

    # print(yhat)

    # Change from 0-9 labels to "one-hot" binary vector labels. For instance, 
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    # ...

    # Train the model
    Wtilde = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, alpha=.1)

    # print(np.shape(Wtilde))
    # print(np.shape(trainingImages))
    # print(np.shape(trainingLabels))

    Xtest_tilde = np.column_stack([testingImages, np.ones(testingImages.shape[0])])
    Ztest = Xtest_tilde @ Wtilde
    Yhat_test = softmax(Ztest)
    predictions = np.argmax(Yhat_test, axis=1)

    accuracy = np.mean(predictions == testingLabels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Visualize the vectors
    # ...
