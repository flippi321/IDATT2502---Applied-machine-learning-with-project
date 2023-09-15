import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

matplotlib.rcParams.update({'font.size': 11})

# Read data from csv file and save as torch tensors with 32-bit floats
data = pd.read_csv('Øvning 2/a/data.csv', sep=',', header=0)
x_train = torch.tensor(data[['X1', 'X2']].values, dtype=torch.float32)
y_train = torch.tensor(data['Y'].values, dtype=torch.float32).reshape(-1, 1)

# Create a model for logistic regression
class LogisticRegressionModel:
    def __init__(self):
        self.W = torch.rand((2, 1), requires_grad=True, dtype=torch.float32)
        self.b = torch.rand((1,), requires_grad=True, dtype=torch.float32)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

model = LogisticRegressionModel()
epochs = 100000
learning_rate = 1e-3

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], lr=learning_rate)

for epoch in range(epochs):
    if epoch % 5000 == 0:
        print(f"Epoch: {epoch}")
        print(f"W = {model.W.tolist()}")
        print(f"b = {model.b.tolist()}")
        print(f"loss = {model.loss(x_train, y_train)}")
        print("------------------------------")

    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for the next step

print(f"[0, 0]: {round(float(model.f(x_train[0])), 3)}, loss: {round(float(model.loss(x_train[0], y_train[0])), 3)}")
print(f"[1, 0]: {round(float(model.f(x_train[1])), 3)}, loss: {round(float(model.loss(x_train[1], y_train[1])), 3)}")
print(f"[0, 1]: {round(float(model.f(x_train[2])), 3)}, loss: {round(float(model.loss(x_train[2], y_train[2])), 3)}")
print(f"[1, 1]: {round(float(model.f(x_train[3])), 3)}, loss: {round(float(model.loss(x_train[3], y_train[3])), 3)}")

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for points
x = x_train[:, 0].numpy()
y = x_train[:, 1].numpy()
z1 = y_train.flatten().numpy()
ax.scatter(x, y, z1, color="blue")

# Decision boundary plot
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
zz = model.f(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).reshape(xx.shape).detach().numpy()
ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='viridis')

ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Svar')

plt.show()
