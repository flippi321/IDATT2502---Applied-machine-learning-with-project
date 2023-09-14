import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

matplotlib.rcParams.update({'font.size': 11})

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output


# Create a model for logistic regression
class LogisticRegressionModel:
    def __init__(self):
        self.W = torch.rand((784, 10), requires_grad=True, dtype=torch.float32)
        self.b = torch.rand((10,), requires_grad=True, dtype=torch.float32)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)
    
    # Calculate accuracy
    def accuracy(self, x, y):
        predictions = self.f(x)
        correct_predictions = torch.argmax(predictions, dim=1) == torch.argmax(y, dim=1)
        accuracy = correct_predictions.float().mean()
        return accuracy.item()


model = LogisticRegressionModel()
epochs = 10000
learning_rate = 5e-4

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], lr=learning_rate)

print("------------------------------")
for epoch in range(epochs):
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}")
        print(f"loss = {model.loss(x_train, y_train)}")
        print("------------------------------")

    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for the next step

# Calculate accuracy
print(f"The model has an accuracy of {round(model.accuracy(x_test, y_test), 2)}")

# Show the input of the first ten observations in the training set
print("\n\n-----------------------------------------------------------")
print("                     TESTING MODEL")
print("-----------------------------------------------------------")
for i in range (10, 20):
    plt.imshow(x_test[i, :].reshape(28, 28))

    # Print the classification of the observation in the training set
    print(f"For test set value {i}")
    print(f"We got:       {model.f(x_test[i, :].unsqueeze(0))[0].tolist()}")
    print(f"and expected: {y_test[i, :].tolist()}")
    print("-----------------------------------------------------------")

    # Save the input of the observation in the training set
    #plt.imsave(f'x_train_{i}.png', x_train[i, :].reshape(28, 28))    
    plt.show()