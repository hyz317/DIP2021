import torch
import numpy as np
from run import VAGER
from dataset.dataset import DataSet
import argparse

def judge_train():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_set_dir', type=str, default="../data/training")
    parser.add_argument('--base_dir', type=str, default="../data/base")
    parser.add_argument('--novel_dir', type=str, default="../data/novel")
    args = parser.parse_args()

    dset = DataSet(
        training_img_dir=args.train_set_dir,
        base_info_dir=args.base_dir,
        test_img_dir=args.base_dir
    )

    with open('train.txt', 'r') as fin:
        acc = 0
        preds = fin.read().strip().split('\n')
        for i in range(len(preds)):
            if int(preds[i]) == dset.training_set['labels'][i]:
                acc += 1
        print(acc / len(preds))

def main():
    acc = 0
    with open('labels.txt', 'r') as f1:
        label = f1.read().strip().split('\n')
    with open('proj2_prediction_8.txt', 'r') as f2:
        pred = f2.read().strip().split('\n')[:500]
    for i in range(500):
        if label[i]==pred[i]:
            acc += 1
    print(acc / 500)
    judge_train()
            

if __name__ == '__main__':
    main()