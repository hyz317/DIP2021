from baseline.config import DEVICE
import torch
import torchvision
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class SimpleConv(nn.Module):
    """
    A simple image to vector CNN which takes image of dimension (224 x 224 x 3) and return column vector length 64
    """

    def conv_block(self, in_channels, out_channels=64, kernel_size=3):
        return torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            self.conv_block(3, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        return x


class ProtoNet(nn.Module):

    def __init__(self):
        super(ProtoNet, self).__init__()
        self.f = SimpleConv()

    def random_sample_cls(self, datax, datay, Ns, Nq, cls):
        """
        Randomly samples Ns examples as support set and Nq as Query set
        """
        data = datax[(datay == cls).nonzero()]
        perm = torch.randperm(data.shape[0])
        idx = perm[:Ns]
        S_cls = data[idx]
        idx = perm[Ns: Ns + Nq]
        Q_cls = data[idx]
        return S_cls.to(DEVICE), Q_cls.to(DEVICE)

    def get_centroid(self, S_cls, Nc):
        """
        Returns a centroid vector of support set for a class
        """
        return torch.sum(self.f(S_cls), 0).unsqueeze(1).transpose(0, 1) / Nc

    def get_query_y(self, Qy, Qyc, class_label):
        """
        Returns labeled representation of classes of Query set and a list of labels.
        """
        labels = []
        for i in range(len(Qy)):
            labels += [Qy[i]] * Qyc[i]
        labels = np.array(labels).reshape(len(labels), 1)
        label_encoder = LabelEncoder()
        Query_y = torch.Tensor(label_encoder.fit_transform(labels).astype(int)).long().to(DEVICE)

        Query_y_labels = np.unique(labels)
        return Query_y, Query_y_labels

    def get_centroid_matrix(self, centroid_per_class, Query_y_labels):
        """
        Returns the centroid matrix where each column is a centroid of a class.
        """
        centroid_matrix = torch.Tensor().to(DEVICE)
        for label in Query_y_labels:
            centroid_matrix = torch.cat((centroid_matrix, centroid_per_class[label]))
        return centroid_matrix

    def get_query_x(self, Query_x, centroid_per_class, Query_y_labels):
        """
        Returns distance matrix from each Query image to each centroid.
        """
        centroid_matrix = self.get_centroid_matrix(centroid_per_class, Query_y_labels)
        Query_x = self.f(Query_x)
        m = Query_x.size(0)
        n = centroid_matrix.size(0)
        
        # The below expressions expand both the matrices such that they become compatible to each other in order to caclulate L2 distance.
        # Expanding centroid matrix to "m".
        centroid_matrix = centroid_matrix.expand(m, centroid_matrix.size(0), centroid_matrix.size(1))
        Query_matrix = Query_x.expand(n, Query_x.size(0), Query_x.size(1)).transpose(0, 1)  # Expanding Query matrix "n" times
        Qx = torch.pairwise_distance(centroid_matrix.transpose(1, 2), Query_matrix.transpose(1, 2))
        return Qx
    
    def forward(self, datax, datay, Ns, Nc, Nq, total_classes):
        """
        Implementation of one episode in Prototypical Net
        datax: Training images
        datay: Corresponding labels of datax
        Nc: Number  of classes per episode
        Ns: Number of support data per class
        Nq:  Number of query data per class
        total_classes: Total classes in training set
        """
        k = total_classes.shape[0]
        K = np.random.choice(total_classes, Nc, replace=False)
        Query_x = torch.Tensor().to(DEVICE)
        Query_y = []
        Query_y_count = []
        centroid_per_class = {}
        class_label = {}
        label_encoding = 0

        for cls in K:
            S_cls, Q_cls = self.random_sample_cls(datax, datay, Ns, Nq, cls)
            centroid_per_class[cls] = self.get_centroid(S_cls, Nc)
            class_label[cls] = label_encoding
            label_encoding += 1

            # Joining all the query set together
            Query_x = torch.cat((Query_x, Q_cls), 0)
            Query_y += [cls]
            Query_y_count += [Q_cls.shape[0]]
        
        Query_y, Query_y_labels = self.get_query_y(Query_y, Query_y_count, class_label)
        Query_x = self.get_query_x(Query_x, centroid_per_class, Query_y_labels)
        return Query_x, Query_y


class Finetuning(nn.Module):
    def __init__(self):
        super(Finetuning, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.simple_fc = nn.Linear(4096, 50)
        self.fc = nn.Sequential(
            nn.Linear(4096, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 50),
        )

    def forward(self, x):
        x = self.alexnet.classifier[:6](x)
        # x = self.simple_fc(x)
        x = self.fc(x)
        return x
