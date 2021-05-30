import torch
import sys
sys.path.append('..')
from dataset.dataset import DataSet

# from dataset import DataSet
import numpy as np
import argparse
import logging
from tqdm import tqdm
import torchvision
import torch.nn as nn
# import torchvision.models as models

def L(V, T, W, A):
    a = torch.norm((torch.matmul(V, T) - W), p='fro') ** 2 
    b = torch.norm((A - torch.matmul(V, V.T)), p='fro') ** 2    
    return a + b

class VAGER(nn.Module):
    def __init__(self):
        super().__init__()
        self.V = nn.Linear(1000, 2000, bias=False)
        self.T = nn.Linear(2000, 4096, bias=False)
    
    def forward(self, A, W):
        return torch.norm((torch.matmul(self.V.weight.T, self.T.weight.T) - W), p='fro') ** 2 + torch.norm((A - torch.matmul(self.V.weight.T, self.V.weight)), p='fro') ** 2 


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_set_dir', type=str, default="../data/training")
    parser.add_argument('--base_dir', type=str, default="../data/base")
    parser.add_argument('--novel_dir', type=str, default="../data/novel")
    args = parser.parse_args()

    dataset = DataSet(
        training_img_dir=args.train_set_dir,
        base_info_dir=args.base_dir
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    a = torch.zeros((1001, 4096)).to(device)
    '''
    A = torch.zeros((1001, 1001)).to(device)
    logging.warning(len(dataset.base_info['labels']))
    for i in range(len(dataset.base_info['labels'])):
        a[dataset.base_info['labels'][i]] += dataset.base_info['features'][i]
    for i in range(1, 1001):
        a[i] /= 100

    for i in tqdm(range(1, 1001)):
        for j in range(1, 1001):
            A[i, j] = torch.dot(a[i], a[j]) / (torch.norm(a[i], p=2) * torch.norm(a[j], p=2))
    
    np.save('A.npy', A)
    '''
    A = torch.Tensor(np.load('A.npy')[1:, 1:]).to(device)
    print(A)
    alexnet = torchvision.models.alexnet(pretrained=True)

    W = alexnet.classifier[6].weight.to(device)
    print(type(W))
    print(W.shape)
    init = False
    if init:
        model = VAGER().to(device)
    else:
        model = torch.load('bak.bin')
        print(type(model))
    # loss = L(V, T, W, A)
    
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    loss = nn.L1Loss()
    least = 10000
    for i in range(0, 10000):
        optim.zero_grad()
        output = model(A, W)
        # loss = loss(output, 0)
        loss = output
        print(loss)
        loss.backward()
        optim.step()
        if least > loss:
            least = loss
            torch.save(model, 'vager.bin')

    beta = 1

if __name__ == '__main__':
    main()
