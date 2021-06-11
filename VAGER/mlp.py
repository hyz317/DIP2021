import torch
import torch.nn as nn
import torch.nn.functional as F
from run import VAGER
import numpy as np
from dataset.dataset import DataSet
import random
import argparse

class MLP(nn.Module):
    def __init__(self, weight):
        super().__init__()
        # self.fc = nn.Linear(4096, 50)
        self.fc = nn.Parameter(data=weight, requires_grad=True) # 50 * 4096
        # self.fc.weight = weight
    
    def forward(self, input):
        logits = torch.zeros(50)
        for i in range(50):
            logits[i] = torch.matmul(input, self.fc.data[i])

        return (logits)

def main():
    w_new = torch.Tensor(np.load('w_new.npy'))
    model = MLP(w_new)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_set_dir', type=str, default="../data/training")
    parser.add_argument('--base_dir', type=str, default="../data/base")
    parser.add_argument('--novel_dir', type=str, default="../data/novel")
    args = parser.parse_args()

    dataset = DataSet(
        training_img_dir=args.train_set_dir,
        base_info_dir=args.base_dir,
        test_img_dir=args.base_dir
    )

    train_set = []
    for i in range(len(dataset.training_set['features'])):
        tmp = []
        tmp.append(dataset.training_set['features'][i])
        tmp.append(torch.tensor([dataset.training_set['labels'][i] - 1]))
        train_set.append(tmp)
    random.shuffle(train_set)
    optim = torch.optim.SGD(model.parameters(), lr=0.0001)
    loss = nn.MSELoss()
    print(len(train_set))
    for epoch in range(100):
        for i in range(len(train_set)):
            output = model(train_set[i][0])
            print(train_set[i][0].shape)
            label = train_set[i][1]
            print(label.shape)
            print(output.shape)
            l = loss(output, label)
            print(l)
            l.backward()
            optim.step()

if __name__ == '__main__':
    main()
