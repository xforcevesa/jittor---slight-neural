import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms


class VisionDataset(Dataset):
    def __init__(self, train, dataset="MNIST"):
        transform = transforms.Compose([transforms.ToTensor()])

        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10,
        }

        train_dataset = datasets[dataset](
            "./data", download=True, train=train, transform=transform
        )

        self.dataset_len = len(train_dataset.data)

        if dataset == "MNIST" or dataset == "Fashion":
            pad = transforms.Pad(2)
            data = pad(train_dataset.data)
            train_dataset.data = data
            self.depth = 1
            self.size = 32
        elif dataset == "CIFAR":
            data = torch.Tensor(train_dataset.data)
            train_dataset.data = data
            self.depth = 3
            self.size = 32
        data = ((data / 255.0) * 2.0) - 1.0
        self.input_seq = data
        self.train_dataset = train_dataset

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.train_dataset[item]
    
    def __iter__(self):
        self.iter_index = 0
        return self
    
    def __next__(self):
        item = self[self.iter_index]
        self.iter_index += 1
        return item

