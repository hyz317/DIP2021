import sys
sys.path.append("..")

from datetime import datetime
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary

import config
from model import Finetuning, BasicProtoNet, AlexProtoNet
from dataset.dataset import DataSet
from data import CaltechDataset, FewShotBatchSampler

epochs = 1000
n_train = 5
k_train = 10
q_train = 5
n_test = 1
k_test = 5
q_test = 1
episodes_per_epoch = 100
num_tasks = 1
lr = 1e-3
lr_step_size = 20
lr_gamma = 0.5
model_dict = {'Finetuning' : Finetuning(), 'BasicProtoNet' : BasicProtoNet(), 'AlexProtoNet' : AlexProtoNet()}

def calc_distances(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def prepare_batch(batch, k, q):
    data, label = batch
    x = data.float().to(config.DEVICE)
    y = torch.arange(0, k, 1 / q).long().to(config.DEVICE)
    return x, y

def predict(model, n, k, x, y=None):
    embeddings = model(x)
    supports, queries = embeddings[:n * k], embeddings[n * k:]
    prototypes = supports.reshape(k, n, -1).mean(dim=1)
    distances = calc_distances(queries, prototypes)
    log_p_y = (-distances).log_softmax(dim=1)

    if y is not None and model.loss is not None:
        loss = model.loss(log_p_y, y)
    else:
        loss = None

    y_pred = (-distances).softmax(dim=1)
    return y_pred, loss

def evaluate(model, history, dataloader, n, k, q, epoch_idx):
    model.eval()

    total_loss, total_acc, data_cnt = 0, 0, 0
    with torch.no_grad():
        if config.MODEL == 'Finetuning':
            for i, batch in enumerate(dataloader, 1):
                x, y = batch
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)
                y_pred = model(x)
                loss = model.loss(y_pred, y)
                data_cnt += y_pred.shape[0]
                total_loss += loss.item() * y_pred.shape[0]
                total_acc += torch.eq(y_pred.argmax(dim=-1), y).sum().item()

        else:
            for i, batch in enumerate(dataloader, 1):
                x, y = prepare_batch(batch, k, q)
                y_pred, loss = predict(model, n, k, x, y)
                data_cnt += y_pred.shape[0]
                total_loss += loss.item() * y_pred.shape[0]
                total_acc += torch.eq(y_pred.argmax(dim=-1), y).sum().item()
                

    history['loss'].append(total_loss / data_cnt)
    history['accuracy'].append(total_acc / data_cnt)
    print(f'{datetime.now()} [epoch {epoch_idx} eval] loss: {total_loss / data_cnt}, accuracy: {total_acc / data_cnt}')


def train_epoch(model, optimizer, scheduler, dataloader, n, k, q, epoch_idx):
    model.train()

    if config.MODEL == 'Finetuning':
        for i, batch in enumerate(dataloader, 1):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            y_pred = model(x)
            loss = model.loss(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f'{datetime.now()} [epoch {epoch_idx} batch {i}] loss: {loss.item()}')
    else:
        for i, batch in enumerate(dataloader, 1):
            optimizer.zero_grad()

            x, y = prepare_batch(batch, k, q)
            y_pred, loss = predict(model, n, k, x, y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f'{datetime.now()} [epoch {epoch_idx} batch {i}] loss: {loss.item()}')

def main():

    dataset = DataSet('../training_seamed', '../test_seamed', '../base')
    dataset.training_set['labels'] = dataset.training_set['labels'] - 1 
    dataset.test_set['labels'] = dataset.test_set['labels'] - 1
    train_set = CaltechDataset('train', dataset.training_set)
    if config.MODEL == 'Finetuning':
        train_loader = DataLoader(train_set, num_workers=0, batch_size=32, shuffle=True)
    else:
        train_loader = DataLoader(train_set, num_workers=0, batch_sampler=FewShotBatchSampler(
            train_set, episodes_per_epoch=episodes_per_epoch, n=n_train, k=k_train, q=q_train, num_tasks=num_tasks
        ))

    test_set = CaltechDataset('test', dataset.test_set)

    if config.MODEL == 'Finetuning':
        test_loader = DataLoader(test_set, num_workers=0, batch_size=32)
    else:
        test_loader = DataLoader(test_set, num_workers=0, batch_sampler=FewShotBatchSampler(
            test_set, episodes_per_epoch=episodes_per_epoch, n=n_test, k=k_test, q=q_test, num_tasks=num_tasks
        ))

    model = model_dict[config.MODEL].to(config.DEVICE)

    if config.MODEL == 'Finetuning':
        optimizer = Adam(model.parameters(), lr=1e-5)
        scheduler = None
        history = {'loss': list(), 'accuracy': list()}
    else:
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        history = {'loss': list(), 'accuracy': list()}

    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, scheduler, train_loader, n_train, k_train, q_train, epoch)
        evaluate(model, history, test_loader, n_test, k_test, q_test, epoch)

        # save model and history
        if epoch == 1 or history['accuracy'][-1] > max(history['accuracy'][:-1]):
            torch.save(model.state_dict(), f'{config.MODEL_PATH}/protonets.ckpt')

if __name__ == "__main__":
    main()
