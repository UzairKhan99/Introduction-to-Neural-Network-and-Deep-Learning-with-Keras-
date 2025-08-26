import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(z)


def relu_derivative(z):
    if relu(z) <= 0:
        return 0
    else:
        return 1


z = np.linspace(-10, 10, 400)


sigmoid_grad = sigmoid_derivative(z)
relu_grad = relu_derivative(z)
plt.figure(figsize=(12, 6))

# Plot Sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z), label="Sigmoid Activation", color="b")
plt.plot(z, sigmoid_grad, label="Sigmoid Derivative", color="r", linestyle="--")
plt.title("Sigmoid Activation & Gradient")
plt.xlabel("Input Value (z)")
plt.ylabel("Activation / Gradient")
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label="ReLU Activation", color="g")
plt.plot(z, relu_grad, label="ReLU Derivative", color="r", linestyle="--")
plt.title("ReLU Activation & Gradient")
plt.xlabel("Input Value (z)")
plt.ylabel("Activation / Gradient")
plt.legend()

plt.tight_layout()
plt.show()
