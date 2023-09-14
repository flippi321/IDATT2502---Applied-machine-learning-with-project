import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import torch

matplotlib.rcParams.update({'font.size': 11})

# Read data from csv file and save as as a torch tensors with 32 bit floats
data = pd.read_csv('Øvning1/a/data.csv', sep=',', header=0, engine='python')
x_train = torch.tensor(data["# length"].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data["weight"].values, dtype=torch.float32).reshape(-1, 1)

# Create model for linear regression
class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()
epochs = 100000
learning_rate = 1e-4

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], lr=learning_rate)

for epoch in range(epochs):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label=f'$f(x) = xW+b$')
plt.legend()
plt.show()
