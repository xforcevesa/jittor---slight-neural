import torch
from dataset import VisionDataset
from model import DiffusionModel
from torch.utils.data import DataLoader
import glob
from matplotlib import pyplot as plt
import numpy as np


# Training hyperparameters
diffusion_steps = 1000
dataset_choice = "MNIST"
max_epoch = 10
batch_size = 128

# Loading parameters
load_model = False
load_version_num = 1

train_dataset = VisionDataset(True, dataset_choice)
val_dataset = VisionDataset(False, dataset_choice)

model = DiffusionModel(train_dataset.size*train_dataset.size, diffusion_steps, train_dataset.depth)


def test_dataset():
    for sample, target in train_dataset:
        print(sample.shape, target)
        sample = np.asanyarray(sample.squeeze())
        plt.imshow(sample, "grey")
        plt.show()
        break


def main():
    test_dataset()


if __name__ == '__main__':
    main()
