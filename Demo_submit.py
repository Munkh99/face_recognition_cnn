import glob
import json
import os
import logging.config
import time
import logging
import PIL.Image as Image
import requests
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary


class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), padding=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=2304, out_features=1024)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)


        self.fc3 = nn.Linear(in_features=128, out_features=10)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)


        x = self.fc3(x)
        output = self.logSoftmax(x)

        return output



class MNISTestSet(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.root = path + "/images/"
        self.image_paths = glob.glob(self.root + "*")
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # open the image
        img = Image.open(self.image_paths[idx]).convert("L")  # convert rgb images to grayscale
        img = self.transform(img)

        # get the image name
        img_name = os.path.basename(self.image_paths[idx]).split(".")[0]
        return img, img_name

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)

    return total_loss / total_samples, total_correct / total_samples


# Define the testing loop
def valid(model, test_loader, criterion, device):
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


def submit(results, url="http://coruscant.disi.unitn.it:3001/results/"):
    res = json.dumps(results)
    # print(res)

    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
        print(f"precision is {result['precision']}")
        print(f"recall is {result['recall']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

    return result


def main():
    logging.basicConfig(filename='logs/log.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d -- %(name)s -- %(levelname)s -- %(message)s',

                        datefmt='%d - %m - %Y %H:%M:%S',
                        level=logging.INFO)

    logging.info("###MODEL RUNNING###")

    # Load MNIST dataset
    train_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True)

    valid_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=False)

    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "mps"

    print(f"Using device: {device}")
    model = NeuralNet().to(device)
    ###
    model_temp = NeuralNet()
    summary(model_temp, (1, 28, 28))

    logging.info("\n" + str(model))
    epoch = 7
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    # Train and evaluate the model
    for epoch in range(epoch):
        train_loss, train_acc = train(model, train_data, optimizer, criterion, device)
        valid_loss, valid_acc = valid(model, valid_data, criterion, device)
        logging.info('Epoch {}: Train loss {:.4f}, Train accuracy {:.4f}, Validation loss {:.4f}, Validation accuracy {:.4f}'.format(
                epoch + 1, train_loss, train_acc, valid_loss, valid_acc))
        print(
            'Epoch {}: Train loss {:.4f}, Train accuracy {:.4f}, Validation loss {:.4f}, Validation accuracy {:.4f}'.format(
                epoch + 1, train_loss, train_acc, valid_loss, valid_acc))

    ### test
    data_path = '/data/test/mnist_test_simulation/'
    dataset = MNISTestSet(data_path)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    result = dict()

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            output = model(data)
            predictions = torch.argmax(output, dim=-1)  # take the hard predictions
            for name, pred in zip(labels, predictions):
                assert name not in result.keys(), f"{name} already in the result dictionary"
                result[name] = pred.item()

    end = time.time()
    print("duration: ", end="\t")
    print(end - start)
    logging.info(f"DURATION = {end - start}, DEVICE = {device}")

    query_random_guess = dict()
    query_random_guess['groupname'] = "Capybara"
    query_random_guess["images"] = result
    with open('data.json', 'w') as f:
        json.dump(query_random_guess, f)

    result = submit(query_random_guess)

    logging.info(f"EPOCH = {epoch}, LEARNING RATE = {lr}, accuracy is {result['accuracy']}")
    logging.info(f"\naccuracy is {result['accuracy']} \nprecision is {result['precision']}\n"f"recall is {result['recall']}")


if __name__ == '__main__':
    main()
