from torch.utils.data import Dataset
import numpy as np

class CaltechDataset(Dataset):

    def __init__(self, data, label):
        super(CaltechDataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, id):
        return self.data[id], self.label[id]

    def __len__(self):
        return self.data.shape[0]