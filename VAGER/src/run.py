import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import random
import copy
from tqdm import tqdm

class BaseDataset(torch.utils.data.Dataset):
    # here labels is a list
    # encoding is a list
    def __init__(self, encodings_labels_dict):
        self.encodings = encodings_labels_dict["encodings"]
        self.labels = encodings_labels_dict["labels"]

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        tmp = self.encodings[idx]
        item = {}
        # item['feature'] = torch.tensor(self.encodings[idx])
        if type(self.encodings[idx]) == torch.Tensor:
            item['feature'] = self.encodings[idx].clone().detach()
        else :
            item['feature'] = torch.tensor(self.encodings[idx])
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

    def devide(self, shuffle=True, novel=True):
        if shuffle and novel:
            tr_en = []
            val_en = []
            tr_l = []
            val_l = []
            for i in range(50):
                tr_en += self.encodings[i * 10 : i * 10 + 8]
                val_en += self.encodings[i * 10 + 8 : i * 10 + 10]
                tr_l += self.labels[i * 10 : i * 10 + 8]
                val_l += self.labels[i * 10 + 8 : i * 10 + 10]
            train = BaseDataset(
                {
                    'encodings': tr_en,
                    'labels' : tr_l
                }
            )
            train.shuffle()
            val = BaseDataset(
                {
                    'encodings': val_en,
                    'labels': val_l
                }
            )
            val.shuffle()
        else:
            alpha = int(0.8 * len(self.encodings))
            train = BaseDataset(
                {
                    'encodings': self.encodings[:alpha],
                    'labels' : self.labels[:alpha]
                }
            )
            train.shuffle()
            val = BaseDataset(
                {
                    'encodings': self.encodings[alpha:],
                    'labels': self.labels[alpha:]
                }
            )
            val.shuffle()
        return train, val


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.linear(input))

class AnologyGraph(nn.Module):
    def __init__(self, class_num, embed_dim, feature_dim, matrixA, matrixW, beta=1.0):
        super().__init__()
        # self.V = nn.Parameter(torch.zeros((class_num, embed_dim), requires_grad=True))
        self.V = nn.Linear(class_num, embed_dim, bias=False)
        self.beta = beta
        # print(self.V.weight.T.shape)
        # self.T = nn.Parameter(torch.zeros((embed_dim, feature_dim), requires_grad=True))
        self.T = nn.Linear(embed_dim, feature_dim, bias=False)
        # print(self.T.weight.T.shape)
        assert matrixA.shape == (class_num, class_num)
        assert matrixW.shape == (class_num, feature_dim)
        self.A = matrixA
        self.W = matrixW
        # self.init_weights()

    def forward(self):
        return torch.norm(torch.matmul(self.V.weight.T, self.T.weight.T) - self.W, p='fro') ** 2 + self.beta * torch.norm(self.A - torch.matmul(self.V.weight.T, self.V.weight), p='fro') ** 2
        # return torch.norm(torch.matmul(self.V, self.T) - self.W, p='fro') ** 2 + self.beta * torch.norm(self.A - torch.matmul(self.V, self.V.T), p='fro') ** 2
    
def cos_dist(a, b):
    return torch.matmul(a, b) / (torch.norm(a, p=2) * torch.norm(b, p=2))

def train_VAGER(base_feature):
    assert base_feature.shape == (100000, 4096)
    class_num = 1000
    embed_dim = 600
    feature_dim = 4096
    class_mean = torch.Tensor(1000, 4096)
    '''
    # calculate A
    for i in range(1000):
        class_mean[i] = torch.mean(base_feature[i*100 : i * 100 + 100], dim=0)
    A = torch.Tensor(1000, 1000)

    for i in tqdm(range(1000)):
        for j in range(1000):
            A[i, j] = cos_dist(class_mean[i], class_mean[j])
    torch.save(A, 'Atensor.pt')
    # calculate W
    alexnet = torchvision.models.alexnet(pretrained=True)
    W = alexnet.classifier[6].weight
    print(W.shape)
    torch.save(W, 'Wtensor.pt')
    '''
    A = torch.load('Atensor.pt')
    A.requires_grad = True
    W = torch.load('Wtensor.pt').detach()
    W.requires_grad = True

    init_train = True
    if init_train:
        model = AnologyGraph(
            class_num=class_num,
            embed_dim=embed_dim,
            feature_dim=feature_dim,
            matrixA=A,
            matrixW=W
        )
    else:
        model = torch.load('bak.bin')
    print('=========')
    # for item in model.parameters():
    #     print(item)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss = nn.L1Loss()
    log_epoch = 100
    rcd = 1000000
    model.train()
    for epoch in range(10000):
        output = model.forward()
        # l = loss(output, torch.norm(torch.Tensor(0), p=1))
        # if epoch % log_epoch == log_epoch - 1:
        print(f'[Analogy Graph] training epoch{epoch}, loss = {output}')
        if output < rcd:
            rcd = output
            torch.save(model, 'vager.bin')
        output.backward()
        optimizer.step()
    return model

def inverse(mat):
    return torch.matmul(
        torch.inverse(torch.matmul(mat.T, mat)),
        mat.T
    )

# base_feature needs to be sorted
def trans(base_feature, novel_feature):
    print(base_feature.shape)
    print(novel_feature.shape)
    model = torch.load('bak.bin')
    novel_class_feature = torch.zeros((50, 4096))
    base_class_feature = torch.zeros((1000, 4096))
    novel_class_analogy = torch.zeros((50, 1000)) # a_new
    embedding_class_analogy = torch.zeros((50, 600)) # v_new
    w_analogy = torch.zeros((50, 4096)) # w_new
    V = model.V.weight.T
    T = model.T.weight.T
    for i in tqdm(range(1000)):
        # print(base_feature[i * 100 : i * 100 + 100].shape)
        base_class_feature[i] = torch.mean(base_feature[i * 100 : i * 100 + 100], dim=0)
    for i in tqdm(range(50)):
        novel_class_feature[i] = torch.mean(novel_feature[i * 10 : i * 10 + 8], dim=0)
        for j in range(1000):
            novel_class_analogy[i, j] = cos_dist(novel_class_feature[i], base_class_feature[j])
        embedding_class_analogy[i] = torch.matmul(novel_class_analogy[i], inverse(V.T))
        w_analogy[i] = torch.matmul(embedding_class_analogy[i], T)
    torch.save(w_analogy, 'w_new.pt')

def predict(novel_feature):
    def preposees_function(example_list, label_list):
        assert len(example_list) == len(label_list)
        result = {}
        for idx in range(len(example_list)):
            result[idx] = [label_list[idx], example_list[idx]]
        sorted_result = sorted(result.items(), key=lambda x: x[1][0], reverse=False)
        encodings_label_dict = {}
        encodings_label_dict['encodings'] = [val[1][1] for val in sorted_result]
        encodings_label_dict['labels'] = [val[1][0] for val in sorted_result]
        return encodings_label_dict
    def gen_novel_label():
        a = np.array(range(100)) / 2
        return [int(i) for i in a] 
    
    all_label = [int(i) for i in np.array(range(500)) / 10]
    whole_dict = preposees_function(novel_feature, all_label)
    whole_dataset = BaseDataset(whole_dict)
    tr_set, val_set = whole_dataset.devide(shuffle=True, novel=True)
    
    w_analogy = torch.load('w_new.pt')
    mlp_W = torch.load('mlp_bak.bin').linear.weight
    print(mlp_W.shape)
    novel_feature_list = []
    for i in range(50):
        novel_feature_list += novel_feature[i * 10 + 8: i * 10 + 10]
    novel_label_list = gen_novel_label()
    novel_feature_label_dict = preposees_function(novel_feature_list, novel_label_list)
    eval_dataset = BaseDataset(novel_feature_label_dict)
    eval_dataset.shuffle()
    dataloader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False
    )
    acc = 0
    for d in tqdm(dataloader):
        pred = torch.argmax(torch.matmul(d['feature'].squeeze(), mlp_W.T))
        # print(torch.matmul(d['feature'], w_analogy.T))
        # print(d['feature'].squeeze().shape)
        # print(torch.matmul(d['feature'].squeeze(), w_analogy.T))
        # print(w_analogy.T)
        # label = d['labels']
        # print(f'pred = {pred}, label = {label}')
        if pred == d['labels']:
            acc += 1
    print(f'[Analogy predict] eval acc = {acc / len(dataloader)}')
    
def main():
    def preposees_function(example_list, label_list):
        assert len(example_list) == len(label_list)
        result = {}
        for idx in range(len(example_list)):
            result[idx] = [label_list[idx], example_list[idx]]
        sorted_result = sorted(result.items(), key=lambda x: x[1][0], reverse=False)
        encodings_label_dict = {}
        encodings_label_dict['encodings'] = [val[1][1] for val in sorted_result]
        encodings_label_dict['labels'] = [val[1][0] for val in sorted_result]
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

    base_features_file = os.path.join('..', '..', 'data', 'base', 'base_feature.npy')
    base_labels_file = os.path.join('..', '..', 'data', 'base', 'base_label.txt')
    base_feature_list = read_features(base_features_file) # index is from 0 - 999
    tmp1 = (base_feature_list[0:2])
    base_label_list = read_labels(base_labels_file) # this is from 1 - 1000
    base_feature_label_dict = preposees_function(base_feature_list, base_label_list)

    novel_features_file = os.path.join('..', '..', 'data', 'base', 'novel_feature.npy')
    novel_feature_list = read_features(novel_features_file)
    novel_label_list = gen_novel_label()
    novel_feature_label_dict = preposees_function(novel_feature_list, novel_label_list)
    novel_dataset = BaseDataset(novel_feature_label_dict)
    novel_train, novel_val = novel_dataset.devide(shuffle=True, novel=True)

    base_feature_list = np.stack(base_feature_label_dict['encodings'], axis=0)
    tmp2 = (base_feature_list[700*100:700*100+2])
    print(np.sum(tmp1 - tmp2))
    # train_VAGER(torch.Tensor(base_feature_list))
    # trans(torch.Tensor(base_feature_list), torch.Tensor(novel_feature_list))
    predict(torch.Tensor(novel_feature_list))

    '''    
    print(len(novel_train))
    print(len(novel_val))
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
    print(novel_val[0])

    model = MLP(4096, 50)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.2)
    loss = nn.CrossEntropyLoss()
    eval_epoch = 10
    record_acc = 0
    record_loss = 1000
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
                # print(output)
                l = loss(output, batch['labels'])
                eval_loss += l
                eval_acc += compute_metric(output, batch['labels'])
            print(f'[epoch {epoch}] eval loss = {eval_loss / len(eval_loader)}')
            print(f'[epoch {epoch}] eval acc = {eval_acc / len(eval_loader)}')
            # if record_acc == 0 and record_loss == 1000:
            #     record_acc = eval_acc
            #     record_loss = eval_loss
            if eval_acc > record_acc: # and (eval_loss - record_loss) / record_loss < 0.02:
                record_acc = eval_acc
                record_loss = eval_loss
                print(f'[epoch {epoch}] saving model...')
                torch.save(model, 'mlp.bin')
    '''

if __name__ == '__main__':
    main()