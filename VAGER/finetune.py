import torch
import numpy as np
from run import VAGER
from dataset.dataset import DataSet
import argparse
import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self):
#         self.fc = nn.Linear(4096, 50)
    
#     def forward(self, input):
#         output = self.fc(input)
#         return nn.ReLU(output)

def main():
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
    a = torch.zeros((1001, 4096))
    print(len(dataset.base_info['labels']))
    for i in range(len(dataset.base_info['labels'])):
        a[dataset.base_info['labels'][i]] += dataset.base_info['features'][i]
    for i in range(1, 51):
        a[i] /= 10

    x_new = torch.zeros((51, 4096))
    print(len(dataset.training_set['labels']))
    for i in range(len(dataset.training_set['labels'])):
        x_new[dataset.training_set['labels'][i]] += dataset.training_set['features'][i]
    for i in range(1, 51):
        x_new[i] /= 10
    
    a_new = torch.zeros((50, 1000))
    for i in range(50):
        for j in range(1000):
            a_new[i, j] = torch.matmul(a[j + 1], x_new[i + 1])
    print(a_new)

    model = torch.load('vager.bin')
    V = model.V.weight
    T = model.T.weight
    print(V.shape)
    print(T.shape)
    Vplus = torch.matmul(torch.inverse(torch.matmul(V.T, V)), V.T)
    print(Vplus.shape)
    v_new = torch.zeros((50, 4000))
    w_new = torch.zeros((50, 4096))

    for i in range(50):
        v_new[i] = torch.matmul(a_new[i], Vplus)
    for i in range(50):
        w_new[i] = torch.matmul(v_new[i], T.T)
    print(len(dataset.training_set['features']))

    np.save('w_new.npy', w_new.detach().numpy())

    with open('train.txt', 'w') as fout:
        for i in range(len(dataset.training_set['features'])):
            x_in = dataset.training_set['features'][i]
            logits = torch.zeros(50)
            for i in range(50):
                logits[i] = torch.matmul(x_in, w_new[i])
            preds = torch.argmax(logits) + 1
            fout.write(f'{preds}\n')
    with open('res.txt', 'w') as fout:
        for i in range(len(dataset.test_set['features'])):
            x_in = dataset.test_set['features'][i]
            logits = torch.zeros(50)
            for i in range(50):
                logits[i] = torch.matmul(x_in, w_new[i])
            preds = torch.argmax(logits) + 1
            fout.write(f'{preds}\n')

if __name__ == '__main__':
    main()