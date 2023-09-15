from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# While the test set is of the size 60K, I only go trough a fraction of this
batches = 600
steps = 20
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  
        self.pool2 = nn.MaxPool2d(kernel_size=2) 
        
        self.dense1 = nn.Linear(64 * 7 * 7, 1024)
        self.dense2 = nn.Linear(1024, 10)

    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x)) 
        x = self.pool2(x)

        x = F.relu(self.dense1(x.reshape(-1, 64 * 7 * 7)))
        x = self.dense2(x)
    
        return x


    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())



model = ConvolutionalNeuralNetworkModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)

print(f"The program has {steps} steps of improvement. Their accuracy values are:")
print(f"{(0):02}: {model.accuracy(x_test, y_test):.04f}")
for epoch in range(steps):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print(f"{(epoch+1):02}: {model.accuracy(x_test, y_test):.04f}")
print("Program done :-D\n\n")

# TODO REMOVE
for i in range(5):
    plt.imshow(x_test[i, :].reshape(28, 28))

    # Print the classification of the observation in the training set
    print(f"For test set value {i}")

    # Rounding the model's prediction to the 4th digit
    rounded_prediction = [round(x, 4) for x in model.f(x_test[i, :].unsqueeze(0))[0].tolist()]
    print(f"We got:       {rounded_prediction}")

    # Rounding the actual data to the 4th digit
    rounded_actual = [round(x, 4) for x in y_test[i, :].tolist()]
    print(f"and expected: {rounded_actual}")

    print("-----------------------------------------------------------")
    plt.show()