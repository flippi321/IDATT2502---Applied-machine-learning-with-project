import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

matplotlib.rcParams.update({'font.size': 11})

# Read data from csv file and save as torch tensors with 32-bit floats
data = pd.read_csv('Øvning 2/c/data.csv', sep=',', header=0)
x_train = torch.tensor(data[['X1', 'X2']].values, dtype=torch.float32)
y_train = torch.tensor(data['Y'].values, dtype=torch.float32).reshape(-1, 1)



# Create a model for logistic regression
class LogisticRegressionModel:
    def __init__(self):
        self.W1 = torch.rand((2, 1), requires_grad=True, dtype=torch.float32)
        self.W2 = torch.rand((2, 1), requires_grad=True, dtype=torch.float32)
        self.b1 = torch.rand((1,), requires_grad=True, dtype=torch.float32)
        self.b2 = torch.rand((1,), requires_grad=True, dtype=torch.float32)

    def layer1(self, x):
        return x @ self.W1 + self.b1

    def layer2(self, x):
        return x @ self.W2 + self.b2

    # Predictor
    def f(self, x):
        return self.layer2(torch.sigmoid(self.layer1(x)))

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)

model = LogisticRegressionModel()
epochs = 100000
learning_rate = 1e-1

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W1, model.W2, model.b1, model.b2], lr=learning_rate)

for epoch in range(epochs):
    if epoch % 5000 == 0:
        print(f"Epoch: {epoch}")
        print(f"W1 = {model.W1[:, 0]}")
        print(f"W2 = {model.W2[:, 0]}")
        print(f"b1 = {round(model.b1.item(), 3)}")
        print(f"b2 = {round(model.b2.item(), 3)}")
        print(f"loss = {round(model.loss(x_train, y_train).item(), 6)}")
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
