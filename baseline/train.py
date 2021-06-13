import sys
sys.path.append("..")

import numpy as np
import config

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from data import CaltechDataset
from model import Finetuning, ProtoNet
from dataset_baseline.dataset import DataSet

num_epoch = 100
lr = 1e-4

# Following just for ProtoNet
num_train_episode = 1000
num_val_episode = 200
frame_size = 10
Ns_train = 5
Nc_train = 5
Nq_train = 5
Ns_val = 5
Nc_val = 5
Nq_val = 5

# Following just for finetuning
batch_size = 32

def train_step_proto(model, data, label, Ns, Nc, Nq, optimizer):
    optimizer.zero_grad()
    Qx, Qy = model(data, label, Ns, Nc, Nq, np.unique(label))
    pred = torch.log_softmax(Qx, dim=-1)
    loss = F.nll_loss(pred, Qy)
    loss.backward()
    optimizer.step()
    acc = torch.mean((torch.argmax(pred, 1) == Qy).float())
    return loss, acc

def val_step_proto(model, data, label, Ns, Nc, Nq):
    with torch.no_grad():
        Qx, Qy = model(data, label, Ns, Nc, Nq, np.unique(label))
        pred = torch.log_softmax(Qx, dim=-1)
        loss = F.nll_loss(pred, Qy)
        acc = torch.mean((torch.argmax(pred, 1) == Qy).float())
    return loss, acc

def train_step_finetuning(model, data, label, optimizer):
    optimizer.zero_grad()
    pred = model(data)
    loss = F.cross_entropy(pred, label)
    loss.backward()
    optimizer.step()
    acc = torch.mean((torch.argmax(pred, 1) == label).float())
    return loss, acc

def val_step_finetuning(model, data, label):
    with torch.no_grad():
        pred = model(data)
        loss = F.cross_entropy(pred, label)
        acc = torch.mean((torch.argmax(pred, 1) == label).float())
    return loss, acc

def train_proto():

    # Reading the data
    dataset = DataSet('../training_seamed')
    dataset.training_set['labels'] = dataset.training_set['labels'] - 1
    dataset.val_set['labels'] = dataset.val_set['labels'] - 1

    # Converting input to pytorch Tensor
    train_data = torch.from_numpy(dataset.training_set['imgs']).float().to(config.DEVICE).permute(0, 3, 1, 2)
    train_label = dataset.training_set['labels']
    val_data = torch.from_numpy(dataset.val_set['imgs']).float().to(config.DEVICE).permute(0, 3, 1, 2)
    val_label = dataset.val_set['labels']

    # Priniting the data and label size
    print(train_data.size(), val_data.size())

    # Initializing protonet
    model = ProtoNet().to(config.DEVICE)

    # Using Pretrained Model
    # model.load_state_dict(torch.load(f'{config.MODEL_PATH}/protonets_0.ckpt'))

    # Training loop
    frame_loss = 0
    frame_acc = 0

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):

        for i in range(num_episode):
            loss, acc = train_step_proto(model, train_data, train_label, Ns_train, Nc_train, Nq_train, optimizer)
            frame_loss += loss.data
            frame_acc += acc.data
            if((i + 1) % frame_size == 0):
                print("Frame Number:", ((i + 1) // frame_size), 'Frame Loss: ', frame_loss.data.cpu().numpy().tolist() /
                      frame_size, 'Frame Accuracy:', (frame_acc.data.cpu().numpy().tolist() * 100) / frame_size)
                frame_loss = 0
                frame_acc = 0

        # Save model
        torch.save(model.state_dict(), f'{config.MODEL_PATH}/protonets_{epoch}.ckpt')

        # Validation loop
        avg_val_loss = 0
        avg_val_acc = 0

        for i in range(num_val_episode):
            loss, acc = val_step_proto(model, val_data, val_label, Ns_val, Nc_val, Nq_val)
            avg_val_loss += loss.data
            avg_val_acc += acc.data

        print('Avg Loss: ', avg_val_loss.data.cpu().numpy().tolist() / num_val_episode,
              'Avg Accuracy:', (avg_val_acc.data.cpu().numpy().tolist() * 100) / num_val_episode)

def train_finetunning():

    # Reading the data
    dataset = DataSet('../training_seamed', model="Finetuning")
    dataset.training_set['labels'] = dataset.training_set['labels'] - 1
    dataset.val_set['labels'] = dataset.val_set['labels'] - 1

    # Converting input to pytorch Tensor
    train_data = torch.from_numpy(dataset.training_set['raw_features']).float()
    train_label = torch.from_numpy(dataset.training_set['labels']).long()
    val_data = torch.from_numpy(dataset.val_set['raw_features']).float()
    val_label = torch.from_numpy(dataset.val_set['labels']).long()

    # Priniting the data and label size
    print(train_data.size(), val_data.size())

    # Initializing protonet
    model = Finetuning().to(config.DEVICE)

    # Using Pretrained Model
    # model.load_state_dict(torch.load(f'{config.MODEL_PATH}/finetuning_1.ckpt'))

    # Training loop
    frame_loss = 0
    frame_acc = 0

    optimizer = Adam(model.parameters(), lr=lr)
    train_set = CaltechDataset(train_data, train_label)
    val_set = CaltechDataset(val_data, val_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epoch):

        j = 0
        for i, data in enumerate(train_loader):
            train_data, train_label = data
            train_data = train_data.to(config.DEVICE)
            train_label = train_label.to(config.DEVICE)

            loss, acc = train_step_finetuning(model, train_data, train_label, optimizer)
            print("Iter Number:", (j), 'Loss: ', (loss.data.cpu().numpy().tolist()), 'Accuracy:', (acc.data.cpu().numpy().tolist()))

        # Save model
        torch.save(model.state_dict(), f'{config.MODEL_PATH}/finetuning_{epoch}.ckpt')

        # Validation loop
        avg_val_loss = 0
        avg_val_acc = 0

        j = 0
        for i, data in enumerate(val_loader):
            val_data, val_label = data
            val_data = val_data.to(config.DEVICE)
            val_label = val_label.to(config.DEVICE)

            loss, acc = val_step_finetuning(model, val_data, val_label)
            avg_val_loss += loss.data
            avg_val_acc += acc.data
            j += 1

        print('Avg Loss: ', avg_val_loss.data.cpu().numpy().tolist() / j,
              'Avg Accuracy:', (avg_val_acc.data.cpu().numpy().tolist() * 100) / j)

def main():
    train_finetunning()
    # train_proto()

if __name__ == "__main__":
    main()
