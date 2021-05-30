import numpy as np
import random
import cv2
import csv
import os

import torch
import torchvision
from torchvision import transforms

class DataSet:
    # self.training_set: dict, "labels" -> [1000], "imgs" -> "[1000]"
    # self.base_info:    dict, "labels" -> [100000], "features" -> [100000 4096]
    # self.baselabelid2name: dict

    
    def __init__(
        self,
        training_img_dir=None,
        test_img_dir=None,
        base_info_dir=None
    ):
        # self.alexnet = torchvision.models.alexnet(pretrained=True).cuda()
        if training_img_dir:
            self.loadTrainingSet(training_img_dir)
        if base_info_dir:
            self.loadBaseInfo(base_info_dir)
        if test_img_dir:
            self.loadTestSet(test_img_dir)

    
    # def img2features(self, imgs):
    #     transform = torchvision.transforms.Compose([
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225]
    #         )
    #     ])
    #     x = torch.tensor(imgs.transpose([0, 3, 1, 2]) / 255., dtype=torch.float32)
    #     ls = []
    #     for i in x:
    #         ls.append(transform(i))
    #     x = torch.stack(ls, 0).cuda()
    #     x = self.alexnet.features(x)
    #     x = self.alexnet.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.alexnet.classifier[:6](x).cpu().detach().numpy()
    #     return x


    def loadTrainingSet(self, path):
        self.training_set = {
            "labels": [],
            "features": []
        }
        imgs = []
        for folder in sorted(os.listdir(path)):
            label = int(folder.split('.')[0])
            for filename in os.listdir(os.path.join(path, folder)):
                img = cv2.imread(os.path.join(path, folder, filename))
                self.training_set["labels"].append(label)
                imgs.append(img)

        self.training_set["labels"] = np.stack(self.training_set["labels"], 0)
        # self.training_set["features"] = self.img2features(np.stack(imgs, 0))
        self.training_set["features"] = torch.tensor(np.load("../data/base/novel_feature.npy"))


    def loadTestSet(self, path):
        self.test_set = {
            "features": [],
            "names": []
        }
        # imgs = []
        # for filename in os.listdir(os.path.join(path)):
        #     img = cv2.imread(os.path.join(path, filename))
        #     imgs.append(img)
        #     self.test_set["names"].append(filename)

        # self.test_set["features"] = self.img2features(np.stack(imgs, 0))
        with open("data/base/names.txt", 'r') as f:
            self.test_set["names"] = f.read().split(" ")
        self.test_set["features"] = torch.tensor(np.load("../data/base/test_feature.npy"))


    def loadBaseInfo(self, path):
        features = np.load(os.path.join(path, "base_feature.npy"))
        with open(os.path.join(path, "base_label.txt"), 'r') as f:
            labels = np.array([ int(i) for i in f.readlines() ])

        self.base_info = {
            "features": features,
            "labels": labels
        }
        self.baselabelid2name = {}
        
        def removeStartBlank(s):
            while s[0] == ' ':
                s = s[1:]
            return s

        with open(os.path.join(path, "base_label_meaning.txt"), 'r') as f:
            f_csv = csv.reader(f)
            cnt = 0
            for row in f_csv:
                cnt += 1
                self.baselabelid2name[cnt] = [ removeStartBlank(i) for i in row ]

    
    def getBaseSubset(self, class_, positive_num, negative_num):
        sub_labels, sub_features = [], []

        # positive samples
        start = list(self.base_info["labels"]).index(class_)
        samples = random.sample(range(100), positive_num)
        for offset in samples:
            sub_labels.append(1)
            sub_features.append(self.base_info["features"][start+offset])

        # negative samples
        samples = random.sample(range(100000), negative_num)
        for index in samples:
            sub_labels.append(-1)
            sub_features.append(self.base_info["features"][index])

        sub_labels = np.array(sub_labels)
        sub_features = np.array(sub_features)
        return torch.tensor(sub_features, dtype=torch.float32), torch.tensor(sub_labels, dtype=torch.float32)


    def getNovelSubset(self, class_, positive_num, negative_num):
        sub_labels, sub_features = [], []

        # positive samples
        start = list(self.training_set["labels"]).index(class_)
        samples = [ i+start for i in random.sample(range(10), positive_num)]
        for offset in samples:
            sub_labels.append(1)
        
        sub_features = self.training_set["features"][samples]

        # negative samples
        samples = random.sample(range(500), negative_num)
        for index in samples:
            sub_labels.append(-1)
        sub_features = torch.cat([sub_features, self.training_set["features"][samples]], 0)

        sub_labels = np.array(sub_labels)
        return sub_features.cpu().clone().detach(), torch.tensor(sub_labels, dtype=torch.float32)