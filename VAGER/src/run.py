import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import random

class BaseDataset(torch.utils.data.Dataset):
    # here labels is a list
    # encoding is a list
    def __init__(self, encodings_labels_dict):
        self.encodings = encodings_labels_dict["encodings"]
        self.labels = encodings_labels_dict["labels"]

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        tmp = self.encodings[idx]
        item = {'feature': torch.tensor(self.encodings[idx])}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def shuffle(self):
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.encodings)
        random.seed(randnum)
        random.shuffle(self.labels)

    def devide(self):
        self.shuffle()
        alpha = int(0.8 * len(self.encodings))
        train = BaseDataset(
            {
                'encodings': self.encodings[:alpha],
                'labels' : self.labels[:alpha]
            }
        )
        val = BaseDataset(
            {
                'encodings': self.encodings[alpha:],
                'labels': self.labels[alpha:]
            }
        )
        return train, val


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.linear(input))

def main():
    def preposees_function(example_list, label_list):
        assert len(example_list) == len(label_list)
        result = {}
        for idx in range(len(example_list)):
            result[idx] = [label_list[idx], example_list[idx]]
        sorted_result = sorted(result.items(), key=lambda x: x[1][0], reverse=False)
        # print(sorted_result[0])
        encodings_label_dict = {}
        encodings_label_dict['encodings'] = [val[1][1] for val in sorted_result]
        encodings_label_dict['labels'] = [val[1][0] for val in sorted_result]
        # print(encodings_label_dict['encodings'][0:10])
        # print(encodings_label_dict['labels'][0:10])
        return encodings_label_dict
    def read_labels(filename):
        with open(filename, 'r') as fin:
            label_list = fin.read().strip().split('\n')
            return np.array(label_list).astype(np.int)
    def read_features(filename):
        a = np.load(filename)
        # print(type(a))
        assert type(a) == np.ndarray
        return np.load(filename)
    def gen_novel_label():
        a = np.array(range(500)) / 10
        return [int(i) for i in a] 
    def compute_metric(logits, labels):
        preds = torch.argmax(logits, dim=1)
        return (preds==labels).numpy().astype(np.float32).mean()

    '''
    base_features_file = os.path.join('..', '..', 'data', 'base', 'base_feature.npy')
    base_labels_file = os.path.join('..', '..', 'data', 'base', 'base_label.txt')
    base_feature_list = read_features(base_features_file)
    base_label_list = read_labels(base_labels_file)
    base_feature_label_dict = preposees_function(base_feature_list, base_label_list)
    '''
    novel_features_file = os.path.join('..', '..', 'data', 'base', 'novel_feature.npy')
    novel_feature_list = read_features(novel_features_file)
    novel_label_list = gen_novel_label()
    novel_feature_label_dict = preposees_function(novel_feature_list, novel_label_list)

    # base_dataset = BaseDataset(base_feature_label_dict)
    novel_dataset = BaseDataset(novel_feature_label_dict)
    novel_train, novel_val = novel_dataset.devide()
    train_loader = DataLoader(
        dataset=novel_train,
        batch_size=4,
        shuffle=False, # already shuffled
    )
    eval_loader = DataLoader(
        dataset=novel_val,
        batch_size=4,
        shuffle=False, # already shuffled
    )
    print((novel_train[0]))

    model = MLP(4096, 50)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.2)
    loss = nn.CrossEntropyLoss()
    eval_epoch = 10
    for epoch in range(200):
        total_loss = 0
        model.train()
        for batch in train_loader:
            output = model(batch['feature'])
            l = loss(output, batch['labels'])
            total_loss += l
            l.backward()
            optimizer.step()
        print(f'[epoch {epoch}] training loss = {total_loss / len(train_loader)}')
        if epoch % eval_epoch == eval_epoch - 1:
            eval_loss = 0
            eval_acc = 0
            model.eval()
            for batch in eval_loader:
                output = model(batch['feature'])
                l = loss(output, batch['labels'])
                eval_loss += l
                eval_acc += compute_metric(output, batch['labels'])
            print(f'[epoch {epoch}] eval loss = {eval_loss / len(eval_loader)}')
            print(f'[epoch {epoch}] eval acc = {eval_acc / len(eval_loader)}')


if __name__ == '__main__':
    main()