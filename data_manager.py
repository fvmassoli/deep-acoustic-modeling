import pandas as pd
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class DataManager(object):
    def __init__(self,  dataframes, names, train_mode, batch_sizes, pin_memory):
        self.dataframes = dataframes
        self.names = names
        self.batch_sizes = batch_sizes
        self.pin_memory = pin_memory
        self.train_mode = train_mode
        self.datasets = self._init_datasets()
        if self.train_mode:
            self.weights = self._init_sampler_weights()
        self.loaders = self._init_loaders()

    def _init_datasets(self):
        return {name: (CustomDataset(self.dataframes[name], name)) for name in self.names}

    def _init_sampler_weights(self):
        syllables = [i['class_'] for idx, i in self.dataframes['train'].iterrows()]
        syllables_counts = Counter(syllables)
        ntotal = len(self.dataframes['train'])
        return [(ntotal / syllables_counts[a]) for a in syllables]

    def get_datasets(self):
        return self.datasets

    def _init_loaders(self):
        if self.train_mode:
            train_loader = DataLoader(self.datasets['train'],
                                      sampler=WeightedRandomSampler(self.weights, len(self.weights)),
                                      batch_size=self.batch_sizes['train'],
                                      pin_memory=self.pin_memory)
            val_loader = DataLoader(self.datasets['val'], batch_size=self.batch_sizes['val'], pin_memory=self.pin_memory)
        test_loader = DataLoader(self.datasets['test'], batch_size=self.batch_sizes['test'], pin_memory=self.pin_memory)
        print("Loaded data:"
              "\n\t Train loader: {}"
              "\n\t Val loader:   {}"
              "\n\t Test loader:  {}".format(len(train_loader) if self.train_mode else 0,
                                             len(val_loader) if self.train_mode else 0,
                                             len(test_loader)))
        return dict(train=train_loader, val=val_loader, test=test_loader) if self.train_mode else dict(test=test_loader)

    def get_loaders(self):
        return self.loaders


class CustomDataset(Dataset):
    def __init__(self, dataframe, name):
        self.dataframe = dataframe
        self.name = name
        self.samples = []
        self.class_codec = LabelEncoder()
        self._init_dataset()

    def __getitem__(self, item):
        features, class_, label = self.samples[item]
        return torch.tensor(features).float(), torch.tensor(self.to_one_hot(class_), dtype=torch.long), torch.tensor(
            [label])

    def __len__(self):
        return len(self.samples)

    def _init_dataset(self):
        classes = set()
        for idx, row in tqdm(self.dataframe.iterrows(),
                             total=len(self.dataframe),
                             desc='Constructing {} dataset'.format(self.name),
                             leave=False):
            classes.add(row['path'].split('/')[-2])
            self.samples.append((pd.read_csv(row['path']).to_numpy(), row['class_'], row['label']))
        self.class_codec.fit(list(classes))

    def to_one_hot(self, class_):
        value_idxs = self.class_codec.transform([class_])
        return torch.eye(len(self.class_codec.classes_))[value_idxs]
