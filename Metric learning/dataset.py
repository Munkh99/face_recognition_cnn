import glob
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import random
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, root_dir, label_file, num_classes, transform=None):
        self.out_classes = None
        self.grouped_examples = None
        self.label_file = label_file
        self.image_dir = root_dir
        self.selected_classes = None

        transform_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        transform = transforms.Compose([
            transform_rgb,
            transforms.Resize((224, 224), antialias=True),  # Adjust the desired size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.num_classes = num_classes
        self.transform = transform
        # self.data = self._load_data()

        self.group_examples()

    # def group_examples(self,total_samples=5000):
    #     df = pd.read_csv(self.label_file, delimiter=' ')
    #     grouped = df.groupby('label')
    #     self.grouped_examples = {}
    #     for group, group_df in grouped:
    #         group_name = group
    #         images = group_df["name"].tolist()
    #         # self.grouped_examples[group_name] = images
    #         sampled_images = random.sample(images, num_samples_per_class)
    #         self.grouped_examples[group_name] = sampled_images
    #     self.out_classes = len(self.grouped_examples.keys())
    # def group_examples(self):
    #     df = pd.read_csv(self.label_file, delimiter=' ')
    #     grouped = df.groupby('label')
    #     self.grouped_examples = {}
    #     num_classes = len(grouped)
    #     samples_per_class = self.total_samples // num_classes
    #
    #     for group, group_df in grouped:
    #         group_name = group
    #         images = group_df["name"].tolist()
    #         num_images = len(images)
    #
    #         if num_images < samples_per_class:
    #             additional_samples = samples_per_class - num_images
    #             additional_images = random.choices(images, k=additional_samples)
    #             images.extend(additional_images)
    #
    #         sampled_images = random.sample(images, samples_per_class)
    #         self.grouped_examples[group_name] = sampled_images
    #
    #     self.out_classes = len(self.grouped_examples.keys())

    def group_examples(self):
        df = pd.read_csv(self.label_file, delimiter=' ')
        grouped = df.groupby('label')
        all_classes = list(grouped.groups.keys())
        selected_classes = random.sample(all_classes, self.num_classes)
        self.grouped_examples = {}
        self.out_classes = len(selected_classes)
        self.selected_classes = selected_classes

        for class_label in selected_classes:
            images = grouped.get_group(class_label)["name"].tolist()
            self.grouped_examples[class_label] = images
        print(selected_classes)
    def __len__(self):
        values_length = sum(len(value) for value in self.grouped_examples.values())
        return values_length

    def __getitem__(self, index):

        selected_class = random.choice(self.selected_classes)
        random_img_name_1 = random.choice(self.grouped_examples[selected_class])
        image_1 = self._load_image(os.path.join(self.image_dir, random_img_name_1))

        # same class
        if index % 2 == 0:
            random_img_name_2 = random.choice(self.grouped_examples[selected_class])

            # ensure that the index of the second image isn't the same as the first image
            while random_img_name_2 == random_img_name_1:
                random_img_name_2 = random.choice(self.grouped_examples[selected_class])
            image_2 = self._load_image(os.path.join(self.image_dir, random_img_name_2))
            target = torch.tensor(0, dtype=torch.float)

        # different class
        else:
            other_selected_class = random.choice(self.selected_classes)
            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.choice(self.selected_classes)

            random_img_name_2 = random.choice(self.grouped_examples[other_selected_class])
            image_2 = self._load_image(os.path.join(self.image_dir, random_img_name_2))
            target = torch.tensor(1, dtype=torch.float)

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        return image_1, image_2, target
        # return random_img_name_1, random_img_name_2, target

    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")  # Adjust the conversion based on your image format
        return image


def get_data_face_small_train(image_dir, label_file):
    dataset = CustomDataset(image_dir, label_file)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["train_batch_size"], shuffle=True)
    return dataloader


def get_data_face_small_test(image_dir, label_file):
    dataset = CustomDataset(image_dir, label_file)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["train_batch_size"], shuffle=True)
    return dataloader


def get_data(image_dir, label_file, num_of_samples=2000):
    dataset = CustomDataset(image_dir, label_file, num_of_samples)
    batch_size = 128
    batch_size_test = 256

    ll = len(dataset)
    tr_split = int(ll * 0.8)
    ts_split = int(ll * 0.2)
    if tr_split + ts_split != ll:
        ts_split = ts_split + 1
    print(ll, tr_split, ts_split)

    train_set, val_set = torch.utils.data.random_split(dataset, [tr_split, ts_split])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader


class competitionSet(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.root = path
        self.image_paths = glob.glob(self.root + "*.*")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),  # Adjust the desired size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")  # convert rgb images to grayscale
        img_name = os.path.basename(self.image_paths[idx])
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img, img_name


def get_query_and_gallery(path_to_query='/Users/munkhdelger/PycharmProjects/ML_competition/data/query/',
                          path_to_gallery='/Users/munkhdelger/PycharmProjects/ML_competition/data/gallery/',
                          ):
    query_set = competitionSet(path_to_query)
    gallery_set = competitionSet(path_to_gallery)
    return query_set, gallery_set


def main():
    # random.seed(10)
    # image_dir = "/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Training Images"
    # label_file = "/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels_train.txt"
    # train_loader = get_data_face_small_train(image_dir, label_file)
    # image_dir_test = "/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/Final Testing Images"
    # label_file_test = "/Users/munkhdelger/PycharmProjects/ML_competition/data/Face Images/labels_test.txt"
    # test_loader = get_data_face_small_test(image_dir_test, label_file_test)
    #
    # for idx, (img1, img2, target) in enumerate(train_loader):
    #     print(idx)
    # print("-------------------------------")
    # for idx, (img1, img2, target) in enumerate(test_loader):
    #     print(idx)
    # print("end")

    img = "/Users/munkhdelger/PycharmProjects/ML_competition/data/faces/ARCHIVE/img_celeba"
    label = "/Users/munkhdelger/PycharmProjects/ML_competition/data/faces/ARCHIVE/identity_CelebA.txt"
    num_of_samples = 100
    train_loader, test_loader = get_data(img, label, num_of_samples)
    for idx, (img1, img2, target) in enumerate(train_loader):
        print(idx)
    print("---")
    for idx, (img1, img2, target) in enumerate(test_loader):
        print(idx)


if __name__ == '__main__':
    main()
