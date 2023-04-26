import glob
import os
import pandas as pd
import torchvision
import PIL.Image as Image
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import requests
import json
from itertools import islice


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, out_features=16)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output

    def get_feature_layer(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file, delimiter=' ').iloc[:, 1]
        self.image_paths = glob.glob(os.path.join(root_dir, "*.*"))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=(28, 28)),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)  # preprocessed image
        return img, label


class competitionSet(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.root = path
        self.image_paths = glob.glob(self.root + "*.*")
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=(28, 28)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # open the image
        img = Image.open(self.image_paths[idx]).convert("RGB")  # convert rgb images to grayscale
        img = self.transform(img)

        # get the image name
        # img_name = os.path.basename(self.image_paths[idx]).split(".")[0]
        img_name = os.path.basename(self.image_paths[idx])
        return img, img_name


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    for data, target in train_loader:
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


def valuidation(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    return total_loss / total_samples, total_correct / total_samples


def feature_extraction(query_loader, gallery_loader, model):
    feature_query = dict()
    feature_gallery = dict()

    with torch.no_grad():
        for data, names in query_loader:
            # data = data.to(device)
            output = model.get_feature_layer(data).numpy()
            for i, j in zip(names, output):
                feature_query[i] = j
        for data, names in gallery_loader:
            # data = data.to(device)
            output = model.get_feature_layer(data).numpy()
            for i, j in zip(names, output):
                feature_gallery[i] = j

    # for i, j in feature_query.items():
    #     print(i, j)
    # for i, j in feature_gallery.items():
    #     print(i, j)

    return feature_query, feature_gallery


def find_distance(array1, array2):
    return np.linalg.norm(array1 - array2)


def submit(results, url="http://kamino.disi.unitn.it:3001/results/"):
    res = json.dumps(results)
    # print(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


def find_similarity(feature_query, feature_gallery, N):
    result = dict()
    for q_name, q_feature in feature_query.items():
        tmp = dict()
        for g_name, g_feature in feature_gallery.items():
            distance = find_distance(q_feature, g_feature)  # returns distance score
            tmp[g_name] = distance

        tmp = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1], reverse=True)}  # sort tmp by values
        # result[q_name] = list(tmp.keys())

        result[q_name] = take(N, tmp.keys())
    return result


def main():
    train_set = FaceDataset(
        csv_file='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels.txt',
        root_dir='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Training Images')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

    validation_set = FaceDataset(
        csv_file='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels_test.txt',
        root_dir='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Testing Images')
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16)

    path_to_query = '/Users/munkhdelger/PycharmProjects/ML_competition/data/query/'
    path_to_gallery = '/Users/munkhdelger/PycharmProjects/ML_competition/data/gallery/'

    query_set = competitionSet(path_to_query)
    gallery_set = competitionSet(path_to_gallery)

    # data loaders
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=64, shuffle=False)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=64, shuffle=False)

    #
    model = NeuralNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate the model
    for epoch in range(1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = valuidation(model, validation_loader, criterion)

        print('Epoch {}: Train loss {:.4f}, Train accuracy {:.4f}, Test loss {:.4f}, Test accuracy {:.4f}'.format(
            epoch + 1, train_loss, train_acc, test_loss, test_acc))

    # Get feature extractions
    feature_query, feature_gallery = feature_extraction(query_loader, gallery_loader, model)

    # similarity for each query img to feature gallery images
    top_n = 5
    result = find_similarity(feature_query, feature_gallery, top_n)

    # preperation for submit
    query_random_guess = dict()
    query_random_guess['groupname'] = "Capybara"
    query_random_guess["images"] = result
    with open('data.json', 'w') as f:
        json.dump(query_random_guess, f)

    # result = submit(query_random_guess)

if __name__ == '__main__':
    main()

"""
Done:
1. Getting train and validation data to respect DataLoaders
2. Getting query and gallery data to respect Dataloaders
3. To Train and validate the model with simple architecture
4. Running the model with gallery and query images, then getting feature extraction from the model
5. Finding the distance between a query image and gallery image then sorting by distance scores
6. To take top N images
7. Created result dictionary and convert it to JSON for the submit function

To Do:
1. CNN Architecture - Use fine tuned model? 
2. Data set
3. Data preparation 
    - normalization
    - augmentation
    - how to resize pictures with different size, or crop? how to?
    


"""
