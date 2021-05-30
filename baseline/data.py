import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms, datasets
from tqdm import tqdm

import config

class CaltechDataset(Dataset):

    def __init__(self, subset, origin_data):
        super(CaltechDataset, self).__init__()
        if subset not in ('train', 'test'):
            raise(ValueError, 'Subset can only be [train] or [test]')
        self.subset = subset
        self.origin_data = origin_data

        self.df = self.load_data()
        self.idx_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        label = self.idx_to_class_id[item]
        if config.MODEL == 'Finetuning':
            feature = self.origin_data['raw_features'][item]
            return torch.from_numpy(feature), label
        else:
            img = self.origin_data['imgs'][item].transpose(2, 0, 1)
            img = (img - img.min()) / (img.max() - img.min())
            return torch.from_numpy(img), label

    def __len__(self):
        return len(self.df)

    def load_data(self):

        print(f'Loading {self.subset} dataset')
        raw_feature_data = self.origin_data['raw_features']
        features_data = self.origin_data['features']
        imgs_data = self.origin_data['imgs']
        labels_data = self.origin_data['labels']

        if config.MODEL == 'Finetuning': origin_data = raw_feature_data
        else: origin_data = imgs_data
        
        total_num = origin_data.shape[0]
        
        progress = tqdm(total=total_num)

        data = []
        for i in range(total_num):
            data.append({
                'subset': self.subset,
                'label': labels_data[i],
            })
            progress.update(1)
        progress.close()

        df = pd.DataFrame(data)
        df = df.assign(id=df.index.values)
        df = df.assign(class_id=df['label'])
        return df

class FewShotBatchSampler(Sampler):

    def __init__(self,
                 dataset: Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = None):

        super(FewShotBatchSampler, self).__init__(dataset)
        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.n = n
        self.k = k
        self.q = q
        self.num_tasks = num_tasks

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = list()
            for _ in range(self.num_tasks):

                df = self.dataset.df
                episode_classes = np.random.choice(df['class_id'].unique(), size=self.k, replace=False)
                df = df[df['class_id'].isin(episode_classes)]

                support_set = dict()
                for class_ in episode_classes:
                    n_samples = df[df['class_id'] == class_].sample(self.n)
                    support_set[class_] = n_samples
                    for _, sample in n_samples.iterrows():
                        batch.append(sample['id'])

                for class_ in episode_classes:
                    q_queries = df[(df['class_id'] == class_) & (~df['id'].isin(support_set[class_]['id']))].sample(self.q)
                    for _, query in q_queries.iterrows():
                        batch.append(query['id'])

            yield np.stack(batch)
