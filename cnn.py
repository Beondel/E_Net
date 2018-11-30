import torch
import torch.nn as nn
import torchvision.datasets as dsets
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import math
from data import MNIST


class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    device = torch.device('cpu')

    epochs = 5
    classes = 10
    batch_size = 100
    lr = 0.001

    train_data = MNIST("./MNIST/training/", "./MNIST/train_labels.csv")
    test_data = MNIST("./MNIST/testing/", "./MNIST/test_labels.csv")

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False)

    model = Conv_Net()  # classes = 10 by default

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):

            images = Variable(images.float())
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.data[0])

            if (i+1) % 100 == 0:
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'
                      % (epoch+1, epochs, i+1, len(train_data)//batch_size, loss.data[0]))
