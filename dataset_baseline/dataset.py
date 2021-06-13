import numpy as np
import random
import cv2
import csv
import os

import torch
import torchvision
from torchvision import transforms

class DataSet:
    """
    self.training_set: dict, "labels" -> [1000], "imgs" -> "[1000]"
    self.base_info:    dict, "labels" -> [100000], "features" -> [100000 4096]
    self.baselabelid2name: dict
    """

    def __init__(
        self,
        training_img_dir=None,
        test_img_dir=None,
        base_info_dir=None,
        model="ProtoNet",
    ):
        if model == "ProtoNet":
            if training_img_dir:
                self.loadTrainingSetProto(training_img_dir)
        else:
            self.alexnet = torchvision.models.alexnet(pretrained=True).cuda()
            if training_img_dir:
                self.loadTrainingSetFinetuning(training_img_dir)
            if base_info_dir:
                self.loadBaseInfo(base_info_dir)

    def image_rotate(self, img, angle):
        """
        Image rotation at certain angle. It is used for data augmentation
        """
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def img2features(self, imgs):
        transform = torchvision.transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        x = torch.tensor(imgs.transpose([0, 3, 1, 2]) / 255., dtype=torch.float32)
        ls = []
        for i in x:
            ls.append(transform(i))
        x = torch.stack(ls, 0).cuda()
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.classifier[:6](x)
        return x

    def img2rawfeatures(self, imgs):
        transform = torchvision.transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        x = torch.tensor(imgs.transpose([0, 3, 1, 2]) / 255., dtype=torch.float32)
        ls = []
        for i in x:
            ls.append(transform(i))
        x = torch.stack(ls, 0).cuda()
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def loadTrainingSetProto(self, path):
        self.training_set = {
            "labels": [],
            "features": []
        }
        self.val_set = {
            "labels": [],
            "features": []
        }

        training_imgs = []
        val_imgs = []
        for folder in os.listdir(path):
            label = int(folder.split('.')[0])
            i = 0
            for filename in os.listdir(os.path.join(path, folder)):
                img = cv2.imread(os.path.join(path, folder, filename))
                img90 = self.image_rotate(img, 90)
                img180 = self.image_rotate(img, 180)
                img270 = self.image_rotate(img, 270)
                if i < 5:
                    for j in range(4):
                        self.training_set["labels"].append(label)
                    training_imgs.append(img)
                    training_imgs.append(img90)
                    training_imgs.append(img180)
                    training_imgs.append(img270)
                else:
                    for j in range(4):
                        self.val_set["labels"].append(label)
                    val_imgs.append(img)
                    val_imgs.append(img90)
                    val_imgs.append(img180)
                    val_imgs.append(img270)
                i += 1

        self.training_set["imgs"] = np.stack(training_imgs, 0)
        self.training_set["labels"] = np.stack(self.training_set["labels"], 0)
        # self.training_set["features"] = self.img2features(np.stack(training_imgs, 0)).cpu().detach().numpy()
        # self.training_set["raw_features"] = self.img2rawfeatures(np.stack(training_imgs, 0)).cpu().detach().numpy()

        self.val_set["imgs"] = np.stack(val_imgs, 0)
        self.val_set["labels"] = np.stack(self.val_set["labels"], 0)
        # self.val_set["features"] = self.img2features(np.stack(val_imgs, 0)).cpu().detach().numpy()
        # self.val_set["raw_features"] = self.img2rawfeatures(np.stack(val_imgs, 0)).cpu().detach().numpy()

    def loadTrainingSetFinetuning(self, path):
        self.training_set = {
            "labels": [],
            "features": []
        }
        self.val_set = {
            "labels": [],
            "features": []
        }

        training_imgs = []
        val_imgs = []
        for folder in os.listdir(path):
            label = int(folder.split('.')[0])
            i = 0
            for filename in os.listdir(os.path.join(path, folder)):
                img = cv2.imread(os.path.join(path, folder, filename))
                if i < 8:
                    self.training_set["labels"].append(label)
                    training_imgs.append(img)
                else:
                    self.val_set["labels"].append(label)
                    val_imgs.append(img)
                i += 1

        self.training_set["imgs"] = np.stack(training_imgs, 0)
        self.training_set["labels"] = np.stack(self.training_set["labels"], 0)
        self.training_set["features"] = self.img2features(np.stack(training_imgs, 0)).cpu().detach().numpy()
        self.training_set["raw_features"] = self.img2rawfeatures(np.stack(training_imgs, 0)).cpu().detach().numpy()

        self.val_set["imgs"] = np.stack(val_imgs, 0)
        self.val_set["labels"] = np.stack(self.val_set["labels"], 0)
        self.val_set["features"] = self.img2features(np.stack(val_imgs, 0)).cpu().detach().numpy()
        self.val_set["raw_features"] = self.img2rawfeatures(np.stack(val_imgs, 0)).cpu().detach().numpy()

    def loadTestSetProto(self, path):
        self.test_set = {
            "features": [],
            "labels": []
        }
        imgs = []
        for i in range(1, 501, 1):
            img = cv2.imread(os.path.join(path, f'testing_{i}.jpg'))
            imgs.append(img)
            
        with open(os.path.join(path, "labels.txt")) as f:
            self.test_set["labels"] = f.read().split("\n")
        
        for i in range(len(self.test_set["labels"])):
            self.test_set["labels"][i] = int(self.test_set["labels"][i])

        self.test_set["imgs"] = np.stack(imgs, 0)
        self.test_set["labels"] = np.stack(self.test_set["labels"], 0)
        # self.test_set["features"] = self.img2features(np.stack(imgs, 0)).cpu().detach().numpy()
        # self.test_set["raw_features"] = self.img2rawfeatures(np.stack(imgs, 0)).cpu().detach().numpy()

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
                

