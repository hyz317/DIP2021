import torch
from torch.nn.modules.activation import Softmax
from torch.nn.modules.batchnorm import BatchNorm1d
import torchvision
from torch import nn

import config

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class Finetuning(nn.Module):
    def __init__(self):
        super(Finetuning, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 50)
        self.loss = torch.nn.CrossEntropyLoss()

        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.normal_(self.fc2.weight, 0, 0.005)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        x = self.alexnet.classifier[:6](x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class BasicProtoNet(nn.Module):
    def __init__(self):
        super(BasicProtoNet, self).__init__()
        self.convs = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64),
        )
        self.fc = nn.Linear(4096, 50)
        self.loss = torch.nn.NLLLoss()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return x

class AlexProtoNet(nn.Module):
    def __init__(self):
        super(AlexProtoNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.fc = nn.Linear(4096, 50)
        self.loss = torch.nn.NLLLoss()

    def forward(self, x):
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.classifier[:6](x)
        # x = self.fc(x)
        return x