import numpy as np
import matplotlib.pyplot as plt


########################################################################################################################
# PROBLEM 1
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor(x, y, d):
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
def softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):
    Xtilde = np.column_stack([trainingImages, np.ones(len(trainingImages))])
    Xtilde_test = np.column_stack([testingImages, np.ones(len(testingImages))])

    num_classes = np.unique(trainingLabels).shape[0]

    Wtilde = np.random.normal(0, 1e-5, size=(Xtilde.shape[1], num_classes))

    Y = one_hot(trainingLabels, num_classes)
    Y_test = one_hot(testingLabels, num_classes)

    num_batches = np.shape(Xtilde)[0] // batchSize
    loss_history = []
    E = 50

    print(f"Training set size: {len(Xtilde)}")
    print(f"Test set size: {len(Xtilde_test)}")
    print(f"Batch size: {batchSize}")
    print(f"Number of epochs: {E}")
    print(f"Learning rate: {epsilon}")
    print(f"Regularization: {alpha}\n")

    for epoch in range(E):
        indices = np.random.permutation(len(Xtilde))
        Xtilde_shuffled = Xtilde[indices]
        Y_shuffled = Y[indices]

        epoch_losses = []

        for i in range(num_batches):
            start = i * batchSize
            end = start + batchSize

            X_batch = Xtilde_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            Z = X_batch @ Wtilde
            Yhat = softmax(Z)

            ce_loss = f_ce(Y_batch, Yhat)
            l2_reg = (alpha / (2 * len(X_batch))) * np.sum(Wtilde[:-1] ** 2)
            total_loss = ce_loss + l2_reg
            loss_history.append(total_loss)
            epoch_losses.append(total_loss)

            grad = grad_ce(Wtilde, X_batch, Y_batch, alpha)
            Wtilde -= epsilon * grad

        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_acc = compute_accuracy(Xtilde, trainingLabels, Wtilde)
            test_acc = compute_accuracy(Xtilde_test, testingLabels, Wtilde)
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1}/{E}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    print("\nPlotting training loss for last 20 mini-batches...")
    plt.figure(figsize=(10, 6))
    last_20_losses = loss_history[-20:]
    plt.plot(range(len(last_20_losses)), last_20_losses, 'b-', linewidth=2, marker='o')
    plt.xlabel('Mini-batch (last 20)')
    plt.ylabel('Training Loss')
    plt.title('Training Loss - Last 20 Mini-batches')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_last_20.png', dpi=150, bbox_inches='tight')
    print("Saved: training_loss_last_20.png")

    plt.figure(figsize=(12, 6))
    plt.plot(loss_history, 'b-', alpha=0.7, linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over All Iterations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_full.png', dpi=150, bbox_inches='tight')
    print("Saved: training_loss_full.png")

    final_test_acc = compute_accuracy(Xtilde_test, testingLabels, Wtilde)
    print(f"PC ACCURACY: {final_test_acc:.2f}%")

    return Wtilde


def compute_accuracy(X, Y, W, print_flag=False):
    if print_flag:
        print(np.shape(X))
        print(np.shape(Y))
        print(np.shape(W))

    Z = X @ W
    Yhat = softmax(Z)
    predictions = np.argmax(Yhat, axis=1)
    return 100.0 * np.mean(predictions == Y)


def f_ce(Y, Yhat):
    return -np.mean(np.sum(Y * np.log(Yhat), axis=1))


def grad_ce(W, X, Y, alpha=0.):
    n = X.shape[0]
    Z = X @ W
    Yhat = softmax(Z)
    grad = (1 / n) * X.T @ (Yhat - Y)

    if alpha > 0:
        reg_grad = np.zeros_like(W)
        reg_grad[:-1] = (alpha / n) * W[:-1]
        grad += reg_grad

    return grad


def one_hot(y, num_classes: int):
    return np.eye(num_classes)[y]


def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def visualize_weights(W):
    print("Weight visualizations...")

    weights = W[:-1, :]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i in range(10):
        row = i // 5
        col = i % 5

        w_image = weights[:, i].reshape(28, 28)

        ax = axes[row, col]
        im = ax.imshow(w_image, cmap='RdBu',
                       vmin=-np.max(np.abs(weights)),
                       vmax=np.max(np.abs(weights)))
        ax.set_title(f'Class {i}: {class_names[i]}', fontsize=10)
        ax.axis('off')

    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.tight_layout()
    plt.savefig('weight_visualizations.png', dpi=150, bbox_inches='tight')
    print("Saved: weight_visualizations.png")
    plt.close()


if __name__ == "__main__":
    x = np.array([1, 2, 3])
    y = np.array([1, 4, 9])
    w = trainPolynomialRegressor(x, y, 3)
    yhat = sum(w[i] * 2 ** i for i in range(len(w)))
    print(f"Polynomial test: f(2) = {yhat:.2f} (expected ~4.0)")
    print()

    trainingImages = np.load("fashion_mnist_train_images.npy")
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy")
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    trainingImages = trainingImages / 255.0
    testingImages = testingImages / 255.0

    n_train = trainingImages.shape[0]
    n_test = testingImages.shape[0]
    trainingImages = trainingImages.reshape(n_train, -1)
    testingImages = testingImages.reshape(n_test, -1)

    print(f"After reshaping - Training: {trainingImages.shape}")
    print(f"After reshaping - Testing: {testingImages.shape}")
    print()

    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100,
                          alpha=0.01)

    visualize_weights(W)

