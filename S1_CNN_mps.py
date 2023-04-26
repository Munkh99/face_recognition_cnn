import glob
import os
import pandas as pd
import torchvision
import PIL.Image as Image
from matplotlib import pyplot as plt
import time
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import requests
import json
from itertools import islice
import torchvision.models as models
from torchsummary import summary
import cv2
import logging

class NeuralNet(nn.Module):

    def __init__(self, out):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=6272, out_features=1024)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(in_features=128, out_features=out)
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
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
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
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x




def crop_face(input_image, input_name):
    # convert to grayscale of each frames
    input_image_np = np.array(input_image)
    gray = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2GRAY)

    # read the haarcascade to detect the faces in an image
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # detects faces in the input image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # print('Number of detected faces:', len(faces))

    cropped_face = []
    # loop over all detected faces
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            # To draw a rectangle in a face
            dx = w * 0.1
            dy = h * 0.1
            # print(x, y, w, h)
            cv2.rectangle(input_image_np, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face = input_image_np[y:y + h, x:x + w]
            # cv2.imshow("Cropped Face", face)
            #         cv2.imwrite(f'face{i}.jpg', face)
            #         print(f"face{i}.jpg is saved")
            cropped_face.append(face)

    # display the image with detected faces
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if len(cropped_face) != 0:
        out = Image.fromarray(cropped_face[0])
    else:
        out = input_image
        # print("-", end='')
    return out


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        df = pd.read_csv(csv_file, delimiter=' ')
        df = df.sort_values(by=['name'])
        self.names = df["name"]
        self.labels = df["label"]
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.*")))
        # self.image_paths = os.listdir(os.path.join(root_dir))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((28, 28)),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            # torchvision.transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.labels)

    def get_unique_labels(self):
        s = set()
        for i in self.labels:
            s.add(i)
        return len(s)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        name = self.names[idx]
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = crop_face(img, name)
        img = self.transform(img)  # preprocessed image
        return img, name, label


class competitionSet(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.root = path
        self.image_paths = glob.glob(self.root + "*.*")
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            # torchvision.transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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


def show_image(idx, data, name, target):
    images_vis = torchvision.utils.make_grid(data)
    print(f"Iteration: {idx}, Images:")
    plt.title(f"Class: {target.numpy()} \n Name: {name}")
    plt.imshow(images_vis.permute(1, 2, 0))
    plt.show()


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    for idx, (data, name, target) in enumerate(train_loader):

        # show_image(idx, data, name, target)

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
        # if idx == 20:
        #     break

    return total_loss / total_samples, total_correct / total_samples


def validation(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    with torch.no_grad():
        for idx, (data, name, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            # if idx == 20:
            #     break

    return total_loss / total_samples, total_correct / total_samples


def feature_extraction(query_loader, gallery_loader, model, device):
    feature_query = dict()
    feature_gallery = dict()

    with torch.no_grad():
        for data, names in query_loader:
            data = data.to(device)
            output = model.get_feature_layer(data).cpu().numpy()

            for i, j in zip(names, output):
                feature_query[i] = j
        for data, names in gallery_loader:
            data = data.to(device)
            output = model.get_feature_layer(data).cpu().numpy()
            for i, j in zip(names, output):
                feature_gallery[i] = j
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

        # tmp = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1], reverse=True)}  # sort tmp by values
        tmp = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1])}  # sort tmp by values
        # result[q_name] = list(tmp.keys())

        result[q_name] = take(N, tmp.keys())
    return result


def mainn():
    logging.basicConfig(filename='logs/log_s1.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d -- %(name)s -- %(levelname)s -- %(message)s',

                        datefmt='%d - %m - %Y %H:%M:%S',
                        level=logging.INFO)

    # dataset = FaceDataset(
    #     csv_file='/Users/munkhdelger/PycharmProjects/ML_competition/data/faces/ARCHIVE/identity_CelebA_under_1000.txt',
    #     # root_dir='/Users/munkhdelger/PycharmProjects/ML_competition/data/faces/ARCHIVE/img_celeba_under_1000')
    #     root_dir='data/faces/ARCHIVE/img_celeba_under_1000_cropped')
    # print('dataset length: ', len(dataset))
    # ll = len(dataset)
    # train_split = round(ll * 0.8)
    # validation_split = round(ll - ll * 0.8)
    batch = 64
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_split, validation_split])
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True)
    # validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=False)
    # out_layer = dataset.get_unique_labels()

    path_to_query = '/Users/munkhdelger/PycharmProjects/ML_competition/data/query/'
    path_to_gallery = '/Users/munkhdelger/PycharmProjects/ML_competition/data/gallery/'

    query_set = competitionSet(path_to_query)
    gallery_set = competitionSet(path_to_gallery)

    # data loaders
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=batch, shuffle=False)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=batch, shuffle=False)

    # on small dataset
    train_set = FaceDataset(
        csv_file='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels_train.txt',
        root_dir='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Training Images')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

    validation_set = FaceDataset(
        csv_file='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels_test.txt',
        root_dir='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Testing Images')
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16)
    out_layer = train_set.get_unique_labels()

    print('outlayer length: ', out_layer)
    ###
    model_temp = NeuralNet(out_layer)
    summary(model_temp, (3, 28, 28))
    ###
    # device = "cpu" for windows ->
    device = "mps"
    model = NeuralNet(out_layer).to(device)
    epoch = 40
    lr = 0.0002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    start = time.time()
    tr_a, ts_a = 0, 0
    # Train and evaluate the model
    for epoch in range(epoch):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = validation(model, validation_loader, criterion, device)

        print('Epoch {}: Train loss {:.4f}, Train accuracy {:.4f}, Test loss {:.4f}, Test accuracy {:.4f}'.format(
            epoch + 1, train_loss, train_acc, test_loss, test_acc))
        tr_a, ts_a = round(train_acc, 3), round(test_acc, 3)
    end = time.time()
    print('duration', end - start)

    # Get feature extractions
    feature_query, feature_gallery = feature_extraction(query_loader, gallery_loader, model, device)

    # similarity for each query img to feature gallery images
    top_n = 20
    result = find_similarity(feature_query, feature_gallery, top_n)

    # preparation for submit
    query_random_guess = dict()
    query_random_guess['groupname'] = "Capybara"
    query_random_guess["images"] = result
    with open('data.json', 'w') as f:
        json.dump(query_random_guess, f)

    # result = submit(query_random_guess)
    logging.info(f"EPOCH = {epoch}, LEARNING RATE = {lr}, train and testaccuracy:  {tr_a} ; {ts_a}")

if __name__ == '__main__':

    mainn()
