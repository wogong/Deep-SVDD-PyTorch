from torch.utils.data import Subset
from PIL import Image
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from .outlier_datasets import (load_cifar10_with_outliers,
                               load_cifar100_with_outliers,
                               load_fashion_mnist_with_outliers,
                               load_mnist_with_outliers,
                               load_svhn_with_outliers)
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np


class MNIST(TorchvisionDataset):

    def __init__(self, normal_class, ratio):
        super().__init__()
        transform_train = transforms.Compose([transforms.ToPILImage(), transforms.Resize(28), transforms.ToTensor(), ])
        transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(28), transforms.ToTensor(), ])

        x_train, y_train = load_mnist_with_outliers(normal_class, ratio)

        # random sampling if the number of data is too large
        max_sample_num = 12000
        if x_train.shape[0] > max_sample_num:
            selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
            x_train = x_train[selected, :]
            y_train = y_train[selected]

        self.train_set = trainset_pytorch(train_data=x_train,
                                    train_labels=y_train,
                                    transform=transform_train)
        self.test_set = trainset_pytorch(train_data=x_train,
                                   train_labels=y_train,
                                   transform=transform_test)


class FMNIST(TorchvisionDataset):

    def __init__(self, normal_class, ratio):
        super().__init__()
        transform_train = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), ])
        transform_test = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), ])

        x_train, y_train = load_fashion_mnist_with_outliers(normal_class, ratio)

        # random sampling if the number of data is too large
        max_sample_num = 12000
        if x_train.shape[0] > max_sample_num:
            selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
            x_train = x_train[selected, :]
            y_train = y_train[selected]

        self.train_set = trainset_pytorch(train_data=x_train,
                                          train_labels=y_train,
                                          transform=transform_train)
        self.test_set = trainset_pytorch(train_data=x_train,
                                         train_labels=y_train,
                                         transform=transform_test)


class SVHN(TorchvisionDataset):

    def __init__(self, normal_class, ratio):
        super().__init__()
        transform_train = transforms.Compose([transforms.ToTensor(), ])
        transform_test = transforms.Compose([transforms.ToTensor(), ])

        x_train, y_train = load_svhn_with_outliers(normal_class, ratio)

        # random sampling if the number of data is too large
        max_sample_num = 12000
        if x_train.shape[0] > max_sample_num:
            selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
            x_train = x_train[selected, :]
            y_train = y_train[selected]

        self.train_set = trainset_pytorch(train_data=x_train,
                                          train_labels=y_train,
                                          transform=transform_train)
        self.test_set = trainset_pytorch(train_data=x_train,
                                         train_labels=y_train,
                                         transform=transform_test)


class CIFAR10(TorchvisionDataset):

    def __init__(self, normal_class, ratio):
        super().__init__()
        transform_train = transforms.Compose([transforms.ToTensor(), ])
        transform_test = transforms.Compose([transforms.ToTensor(), ])

        x_train, y_train = load_cifar10_with_outliers(normal_class, ratio)

        # random sampling if the number of data is too large
        max_sample_num = 12000
        if x_train.shape[0] > max_sample_num:
            selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
            x_train = x_train[selected, :]
            y_train = y_train[selected]

        self.train_set = trainset_pytorch(train_data=x_train,
                                          train_labels=y_train,
                                          transform=transform_train)
        self.test_set = trainset_pytorch(train_data=x_train,
                                         train_labels=y_train,
                                         transform=transform_test)


class CIFAR100(TorchvisionDataset):

    def __init__(self, normal_class, ratio):
        super().__init__()
        transform_train = transforms.Compose([transforms.ToTensor(), ])
        transform_test = transforms.Compose([transforms.ToTensor(), ])

        x_train, y_train = load_cifar100_with_outliers(normal_class, ratio)

        # random sampling if the number of data is too large
        max_sample_num = 12000
        if x_train.shape[0] > max_sample_num:
            selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
            x_train = x_train[selected, :]
            y_train = y_train[selected]

        self.train_set = trainset_pytorch(train_data=x_train,
                                          train_labels=y_train,
                                          transform=transform_train)
        self.test_set = trainset_pytorch(train_data=x_train,
                                         train_labels=y_train,
                                         transform=transform_test)


class trainset_pytorch(data.Dataset):
    def __init__(self, train_data, train_labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.train_data = train_data  # ndarray
        self.train_labels = train_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)  # used if the img is [H, W, C] and the dtype is uint8

        if self.transform is not None:
            # img_int = np.uint8(denormalize_minus1_1(img))
            # img = Image.fromarray(img_int)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.train_data)
