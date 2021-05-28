import torch
from torch import nn
import numpy as np


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(4097, 6144)
        self.fc2 = nn.Linear(6144, 5120)
        self.fc3 = nn.Linear(5120, 4097)
        self.fc4 = nn.Linear(4097, 4097)
        # self.bn1 = nn.BatchNorm1d(6144)
        # self.bn2 = nn.BatchNorm1d(5120)
        # self.bn3 = nn.BatchNorm1d(4097)
        # self.bn4 = nn.BatchNorm1d(4097)
        self.W = nn.Parameter(torch.zeros([1, 4097])).cuda()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc4(x)
        # x = self.bn4(x)
        x = nn.functional.leaky_relu(x)
        return x
