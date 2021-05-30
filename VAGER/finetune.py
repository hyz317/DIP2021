import torch
import numpy as np
from run import VAGER
from dataset.dataset import DataSet
import argparse

def main():
    model = torch.load('bak.bin')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_set_dir', type=str, default="../data/training")
    parser.add_argument('--base_dir', type=str, default="../data/base")
    parser.add_argument('--novel_dir', type=str, default="../data/novel")
    args = parser.parse_args()

    dataset = DataSet(
        training_img_dir=args.train_set_dir,
        base_info_dir=args.base_dir
    )
    a = torch.zeros((1001, 4096))
    for i in range(len(dataset.base_info['labels'])):
        a[dataset.base_info['labels'][i]] += dataset.base_info['features'][i]
    for i in range(1, 1001):
        a[i] /= 100 
    a_new = torch.zeros((50, 1000))


if __name__ == '__main__':
    main()