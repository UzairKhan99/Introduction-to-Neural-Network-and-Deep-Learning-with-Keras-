import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # shape (2,4)
d = np.array([[0], [1], [1], [0]]).T  # shape (1,4)


def initialize_network_parameters():
    inputSize = 2
    hiddenSize = 2
    outputSize = 1
    lr = 0.1
    epochs = 180000  # Number of training epochs

    w1 = np.random.rand(hiddenSize, inputSize) * 2 - 1
    b1 = np.random.rand(hiddenSize, 1) * 2 - 1
    w2 = np.random.rand(outputSize, hiddenSize) * 2 - 1
    b2 = np.random.rand(outputSize, 1) * 2 - 1

    return w1, b1, w2, b2, lr, epochs


w1, b1, w2, b2, lr, epochs = initialize_network_parameters()

error_list = []

# Training
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(w1, X) + b1
    a1 = 1 / (1 + np.exp(-z1))  # sigmoid

    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  # sigmoid

    # Error
    error = d - a2

    # Backpropagation
    dz2 = error * (a2 * (1 - a2))
    dz1 = np.dot(w2.T, dz2) * (a1 * (1 - a1))

    # Update weights & biases
    w2 += lr * np.dot(dz2, a1.T)
    b2 += lr * np.sum(dz2, axis=1, keepdims=True)

    w1 += lr * np.dot(dz1, X.T)
    b1 += lr * np.sum(dz1, axis=1, keepdims=True)

    # Track error
    if (epoch + 1) % 10000 == 0:
        print(f"Epoch: {epoch+1}, Avg error: {np.mean(np.abs(error)):.5f}")
    error_list.append(np.mean(np.abs(error)))

# Testing the trained network
z1 = np.dot(w1, X) + b1
a1 = 1 / (1 + np.exp(-z1))
z2 = np.dot(w2, a1) + b2
a2 = 1 / (1 + np.exp(-z2))

# Print results
print("\nFinal output after training:")
print(a2)
print("Ground truth:", d)
print("Error after training:", error)
print("Average error: %0.05f" % np.average(abs(error)))

# Plot error
plt.plot(error_list)
plt.title("Error Curve")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
