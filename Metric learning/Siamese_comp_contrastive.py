from __future__ import print_function
import argparse, random, copy
from pathlib import Path
import glob
import os
import datetime

from itertools import islice
import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights, ResNet18_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms

import dataset


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer.
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        # res_n = torchvision.models.resnet18(weights=None)
        res_n = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # res_n = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for param in res_n.parameters():
            param.requires_grad = False

        for param in res_n.layer4.parameters():
            param.requires_grad = True

        for param in res_n.layer3.parameters():
            param.requires_grad = True

        # Check the requires_grad status of each layer
        for name, param in res_n.named_parameters():
            print(f"Layer: {name}, Requires Gradient: {param.requires_grad}")

        self.resnet = res_n

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        # self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def feed(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        # output = self.sigmoid(output)
        return output

    def forward(self, x1, x2):
        oupt1 = self.feed(x1)
        oupt2 = self.feed(x2)
        return oupt1, oupt2


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.

    total_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1 = images_1.to(device)
        images_2 = images_2.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        oupt1, oupt2 = model(images_1, images_2)
        loss = criterion(oupt1, oupt2, targets)
        loss.backward()
        optimizer.step()

        # Update cumulative loss
        total_loss += loss.item()
        # Calculate accuracy
        # predicted = (oupt1<oupt2 ).float()  # Convert probabilities to binary predictions
        # correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)

        # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    # accuracy = correct / total_samples

    # Print epoch-level statistics
    # print('Epoch: {}\tAverage Loss: {:.6f}\tAccuracy: {:.2f}%'.format(
    #     epoch, avg_loss, accuracy * 100))
    # return avg_loss, accuracy
    return avg_loss


def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0
    total_samples = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            oupt1, oupt2 = model(images_1, images_2)
            test_loss += criterion(oupt1, oupt2, targets).item() * targets.size(0)  # accumulate batch loss
            # pred = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
            # correct += pred.eq(targets).sum().item()
            total_samples += targets.size(0)

    test_loss /= len(test_loader.dataset)
    # accuracy = correct / total_samples
    # return test_loss, accuracy
    return test_loss


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Adjust the conversion based on your image format
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),  # Adjust the desired size
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization values
    ])

    input_tensor = transform(image)
    input_patch = input_tensor.unsqueeze(0)

    return input_patch


def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


def distance_estimator(query_set, gallery_set, model, device, N):
    model.eval()
    result = dict()
    for img1, img1_name in query_set:
        tmp = dict()
        for img2, img2_name in gallery_set:
            img1, img2 = img1.to(device), img2.to(device)
            out1, out2 = model(img1, img2)
            dissim = torch.nn.functional.pairwise_distance(out1, out2)
            tmp[img2_name] = np.round(dissim.item(), 6)
        tmp = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1], reverse=False)}  # sort tmp by values
        print("--------------------------------------------------------------------")
        print(img1_name)
        print(tmp)
        result[img1_name] = take(N, tmp.keys())
    return result


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m  # margin or radius

    def forward(self, y1, y2, flag):
        # flag = 0 means y1 and y2 are supposed to be same
        # flag = 1 means y1 and y2 are supposed to be different

        euc_dist = torch.nn.functional.pairwise_distance(y1, y2)

        loss = torch.mean((1 - flag) * torch.pow(euc_dist, 2) +
                          (flag) * torch.pow(torch.clamp(self.m - euc_dist, min=0.0), 2))

        return loss


def main():
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=14, metavar='N',
    #                     help='number of epochs to train (default: 14)')
    # parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--no-mps', action='store_true', default=False,
    #                     help='disables macOS GPU training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    # args = parser.parse_args()

    torch.manual_seed(1)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("########### DEVICE :", str(device))
    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}
    # if torch.cuda.is_available():
    #     cuda_kwargs = {'num_workers': 1,
    #                    'pin_memory': True,
    #                    'shuffle': True}
    #     train_kwargs.update(cuda_kwargs)
    #     test_kwargs.update(cuda_kwargs)

    checkpoint_path = Path("/Users/munkhdelger/PycharmProjects/ML_competition/checkpoints")

    # image_dir = "/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Training Images"
    # label_file = "/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels_train.txt"
    # train_loader = dataset.get_data_face_small_train(image_dir, label_file)
    # image_dir_test = "/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Testing Images"
    # label_file_test = "/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels_test.txt"
    # test_loader = dataset.get_data_face_small_test(image_dir_test, label_file_test)

    img = "/Users/munkhdelger/PycharmProjects/ML_competition/data/faces/ARCHIVE/img_celeba_under_1000_cropped"
    label = "/Users/munkhdelger/PycharmProjects/ML_competition/data/faces/ARCHIVE/identity_CelebA_under_1000.txt"
    train_loader, test_loader = dataset.get_data(img, label)

    model = SiameseNetwork().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = ContrastiveLoss()
    start = datetime.datetime.now()
    print(start)

    best_val_accuracy = 0
    best_val_loss = float('inf')
    for epoch in range(60):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_loss = test(model, device, test_loader, criterion)
        print(
            'Epoch {}: Train loss {:.4f}, Train acc {:.4f}, Valid loss {:.4f}, Valid acc {:.4f}, Best acc {:.4f}'.format(
                epoch + 1, train_loss, 0, val_loss, 0, best_val_accuracy))

        # scheduler.step()

        # Update the best model so far
        # if val_accuracy >= best_val_accuracy:
        #
        #     day = start.strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
        #     time = start.strftime("%H:%M:%S")  # Format: HH:MM:SS
        #
        #     torch.save(model.state_dict(),
        #                checkpoint_path / f'Siamese_best_{day}{time}_{round(best_val_accuracy, 4)}.pth')
        #     best_val_accuracy = val_accuracy
        # else:
        #     # update the best val loss so far
        #     best_val_loss = val_loss
        #     trigger = 0
    torch.save(model.state_dict(),
               checkpoint_path / f'Siamese_best__.pth')
    end = datetime.datetime.now()
    print('duration', end - start)
    print(
        f"EPOCH = {20}, best accuracy and best loss:  {round(best_val_accuracy, 3)}, {round(best_val_loss, 3)}")
    # --------------------------------------------------------------------------
    # img1 = '/Users/munkhdelger/PycharmProjects/ML_competition/data/faces/ARCHIVE/img_celeba_under_1000_cropped/000023.jpg'
    # img2 = '/Users/munkhdelger/PycharmProjects/ML_competition/data/faces/ARCHIVE/img_celeba_under_1000_cropped/000023.jpg'
    # img3 = '/Users/munkhdelger/PycharmProjects/ML_competition/data/faces/ARCHIVE/img_celeba_under_1000_cropped/000077.jpg'

    # img1 = load_image(img1).to(device)
    # img2 = load_image(img2).to(device)
    # img3 = load_image(img3).to(device)
    # out1, out2 = model(img1, img2)
    # dissim = torch.nn.functional.pairwise_distance(out1, out2)
    # print(f"output of same {np.round(dissim.item(), 6)}")
    #
    # out1, out2 = model(img2, img3)
    # dissim = torch.nn.functional.pairwise_distance(out1, out2)
    # print(f"output of different {np.round(dissim.item(), 6)}")

    # --------------------------------------------------------------------------
    query_set, gallery_set = dataset.get_query_and_gallery()
    result = distance_estimator(query_set, gallery_set, model, device, N=10)
    # print(result)
    print("###############################################################")
    for i, j in result.items():
        print(i, j)


if __name__ == '__main__':
    main()
