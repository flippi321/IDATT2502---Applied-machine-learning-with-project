import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import torch

matplotlib.rcParams.update({'font.size': 11})

# Read data from csv file and save as as a torch tensors with 32 bit floats
data = pd.read_csv('Øvning1/c/data.csv', sep=',', header=0, engine='python')
x_train = torch.tensor(data["# day"].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data["head circumference"].values, dtype=torch.float32).reshape(-1, 1)

# Create model for linear regression
class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.25]], requires_grad=True, dtype=torch.float32)
        self.b = torch.tensor([[-0.6999]], requires_grad=True, dtype=torch.float32)

    def sigm(self, z):
        return 1 / (1 + torch.exp(-z))

    # Predictor
    def f(self, x):
        return 20 * self.sigm((x @ self.W + self.b)) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()
epochs = 50000
learning_rate = 1e-6

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], lr=learning_rate)

for epoch in range(epochs):
    if(epoch % 5000 == 0):
        print(f"Epoch: {epoch}")
        print(f"W = {model.W}")
        print(f"b = {model.b}")
        print(f"loss = {model.loss(x_train, y_train)}")
        print("------------------------------")

    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
# Plot the graph
x1 = torch.linspace(torch.min(x_train), torch.max(x_train), steps=1000).reshape(-1, 1)
plt.plot(x1, model.f(x1).detach(), label='$\\hat y = f(x) = 20\\sigma(xW+b)+31$')
plt.legend()
plt.show()
