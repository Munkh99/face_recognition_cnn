import logging
import time

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import glob
import os
import pandas as pd
import PIL.Image as Image
from torch.utils.data import DataLoader
from torchsummary import summary
from pathlib import Path


class VGG(nn.Module):
    def __init__(self, out):
        super(VGG, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        vgg16.classifier.append(nn.ReLU())
        vgg16.classifier.append(nn.Linear(in_features=1000, out_features=out))

        for i, param in enumerate(vgg16.features.parameters()):
            if i < 10:
                param.requires_grad = False

        self.base = vgg16
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        x = self.base(x)
        output = self.logSoftmax(x)

        return output

    def get_feature_layer(self, x):
        x = self.base(x)
        # x = self.fc1(x)
        return x


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
            torchvision.transforms.Resize((224, 224)),
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
        # img = crop_face(img, name)
        img = self.transform(img)  # preprocessed image
        return img, name, label


def get_data_face_small(batch_size):
    train_set = FaceDataset(
        csv_file='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels_train.txt',
        root_dir='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Training Images')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

    validation_set = FaceDataset(
        csv_file='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels_test.txt',
        root_dir='/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Testing Images')
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16)
    out_layer = train_set.get_unique_labels()
    return train_loader, validation_loader, out_layer


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


# 0.0004 98.44
def main(batch_size=64, learning_rate=0.0002, num_epochs=80, device="mps", run_name="VGG16"):
    logging.basicConfig(filename='logs/log_vgg16.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d -- %(name)s -- %(levelname)s -- %(message)s',
                        datefmt='%d - %m - %Y %H:%M:%S',
                        level=logging.INFO)

    checkpoint_path = Path("./checkpoints")
    checkpoint_path = checkpoint_path / run_name
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    patience = 3
    trigger = 0

    # Add the last fully connected layer with a new one for face recognition
    # Define the VGG16 model with pre-trained weight
    # weights=VGG16_Weights.IMAGENET1K_V1
    # #weights=VGG16_Weights.DEFAULT

    vgg16 = VGG(16)
    print(summary(vgg16, (3, 224, 224)))
    print(vgg16)

    # Freeze all the pre-trained layers


    train_loader, validation_loader, out_layer = get_data_face_small(batch_size)

    vgg16 = vgg16.to(device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vgg16.parameters(), lr=learning_rate)

    start = time.time()
    best_val_accuracy = 0
    best_val_loss = float('inf')

    # Train the model
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(vgg16, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validation(vgg16, validation_loader, criterion, device)

        print(
            'Epoch {}: Train loss {:.4f}, Train acc {:.4f}, Valid loss {:.4f}, Valid acc {:.4f}, Best acc {:.4f}'.format(
                epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, best_val_accuracy))

        # Update the best model so far
        if val_accuracy >= best_val_accuracy:
            torch.save(vgg16.state_dict(), checkpoint_path / f'best.pth')
            best_val_accuracy = val_accuracy

        # Early Stopping
        # if val_loss > best_val_loss:
        #     trigger += 1
        #     if trigger == patience:
        #         print(f"Validation Accuracy did not imporove for {patience} epochs. Killing the training...")
        #         break
        else:
            # update the best val loss so far
            best_val_loss = val_loss
            trigger = 0
        # ===========================================
    end = time.time()
    print('duration', end - start)
    logging.info(
        f"EPOCH = {num_epochs}, LEARNING RATE = {learning_rate}, best train and best loss:  {round(best_val_accuracy, 3)}, {round(best_val_loss, 3)}")


if __name__ == '__main__':

    main()
