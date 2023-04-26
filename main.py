import time

import torch
import torch.nn as nn
import torchvision
import random
import glob

# Define a simple neural network with a dropout layer
# class NeuralNet(nn.Module):
#     def __init__(self):
#         super(NeuralNet, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(256, 10)
#
#     def forward(self, x):
#         # print("B", x.shape)
#         x = torch.flatten(x, 1)
#         # print("A", x.shape)
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=2880, out_features=500)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=500, out_features=10)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output


# Define the training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(data.shape, output.shape, target.shape)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)

    return total_loss / total_samples, total_correct / total_samples


# Define the testing loop
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    # Load MNIST dataset
    train_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True)

    test_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=False)
    # print(train_data)
    device = "mps" # cpu if windows, cuda if windows&gpu
    device = torch.device(device)
    print(f"Using device: {device}")

    # Initialize the model, optimizer and loss function
    model = NeuralNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start = time.time()

    # Train and evaluate the model
    for epoch in range(10):
        train_loss, train_acc = train(model, train_data, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_data, criterion, device)

        print('Epoch {}: Train loss {:.4f}, Train accuracy {:.4f}, Test loss {:.4f}, Test accuracy {:.4f}'.format(
            epoch + 1, train_loss, train_acc, test_loss, test_acc))

        # print('Epoch {}: Train loss {:.4f}, Train accuracy {:.4f}'.format(
        #     epoch + 1, train_loss, train_acc))

    end = start - time.time()
    print("Duration of epoch iteration ", end)


if __name__ == '__main__':
    main()
