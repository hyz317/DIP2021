import numpy as np
import random
import cv2
import csv
import os


class DataSet:
    # self.training_set: dict, "labels" -> [1000], "imgs" -> "[1000]"
    # self.base_info:    dict, "labels" -> [100000], "features" -> [100000 4096]
    # self.baselabelid2name: dict


    def __init__(
        self,
        training_img_dir=None,
        base_info_dir=None
    ):
        if training_img_dir:
            self.loadTrainingSet(training_img_dir)
        if base_info_dir:
            self.loadBaseInfo(base_info_dir)


    def loadTrainingSet(self, path):
        self.training_set = {
            "labels": [],
            "imgs": []
        }
        for folder in sorted(os.listdir(path)):
            label = int(folder.split('.')[0])
            for filename in os.listdir(os.path.join(path, folder)):
                img = cv2.imread(os.path.join(path, folder, filename))
                self.training_set["labels"].append(label)
                self.training_set["imgs"].append(img)

        self.training_set["labels"] = np.array(self.training_set["labels"])


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


    def shuffle(self):
        length = len(self.training_set["labels"])
        ids = random.sample(range(length), length)
        self.training_set["labels"] = np.array([ self.training_set["labels"][i] for i in ids ])
        self.training_set["imgs"] = [ self.training_set["imgs"][i] for i in ids ]
                

