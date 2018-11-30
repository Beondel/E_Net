import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MNIST(Dataset):
    def __init__(self, images, labels):
        """
            images: root directory for all image csv's
            labels: csv with labels corresponding to each image
                    where labels[i] corresponds to images/i.csv
        """
        self.images = images  # string
        self.labels = np.genfromtxt(labels, delimiter=',')  # array

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = np.genfromtxt(self.images + str(i) + ".csv", delimiter=',')
        label = self.labels[i]
        return (image, label)
