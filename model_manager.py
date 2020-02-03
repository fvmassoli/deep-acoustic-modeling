import os
import numpy as np

import torch
import torch.nn as nn


class ModelManager(object):
    def __init__(self, n_classes, architecture, out_channels, hidden, bidir,
                 dropout, windows, load_ckt, ckt_path, device):
        self.n_classes = n_classes
        self.architecture = architecture
        self.in_channels = 39
        self.out_channels = out_channels
        self.hidden = hidden
        self.bidir = bidir
        self.dropout = dropout
        self.windows = windows
        self.load_ckt = load_ckt
        self.ckt_path = ckt_path
        self.device = device
        self.model = self._init_model()

    def _init_model(self):
        model = Model(self.n_classes, self.in_channels, self.out_channels, self.hidden, self.dropout,
                      self.bidir, self.architecture, self.windows)
        print("Model properties: "
              "\n\t device:       {}"
              "\n\t n_classes:    {}"
              "\n\t in_channels:  {}"
              "\n\t out_channels: {}"
              "\n\t hidden size:  {}"
              "\n\t bidir:        {}"
              "\n\t dropout:      {}"
              "\n\t architecture: {}"
              "\n\t windows:      {}".format(self.device, self.n_classes, self.in_channels, self.out_channels,
                                             self.hidden, self.bidir, self.dropout, self.architecture, self.windows))
        if self.load_ckt:
            self.load_model_ckp(model)
        print(model)
        return model.to(self.device)

    def load_model_ckp(self, model):
        ckt = torch.load(self.ckt_path, map_location=self.device)
        print('\t Loading model from: {}'
              '\n\t With accuracy: {:.2f}'.format(self.ckt_path.split('/')[-2], ckt['acc']))
        model.load_state_dict(ckt['model'])

    def save_best_model(self, optimizer, accuracy, mname):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': accuracy
        }, mname)
        print('\t Model saved at: {}'.format(mname))

    def get_model_params(self):
        return self.model.parameters()

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def forward(self, feat, label, criterion):
        feat = feat.to(self.device)
        label = label.to(self.device) if label is not None else None
        output = self.model(feat)
        loss = criterion(output, label.squeeze(0)) if label is not None else 0.0
        return output, loss

    def get_model(self):
        return self.model

class Model(nn.Module):
    def __init__(self, n_classes, in_channels, out_channels, hidden, dropout, bidir, architecture, windows):
        super(Model, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.bidir = bidir
        self.hidden = hidden
        self.windows = windows
        self.architecture = architecture
        self.mlp = True if self.architecture == 'mlp' else False

        if self.mlp:
            out_f = self.out_channels
            stride = 1
            out_f //= len(self.windows)

            self.conv = nn.ModuleList([nn.Conv1d(in_channels=self.in_channels,
                                                 out_channels=out_f,
                                                 kernel_size=(2 * w + 1),
                                                 padding=w,
                                                 stride=stride)
                                       for w in self.windows])
            avg_size = 20
            self.avg = nn.AdaptiveAvgPool1d(avg_size)
            self.mlp = nn.Sequential(nn.Linear(in_features=out_f*len(self.windows)*avg_size, out_features=self.hidden),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout))
        else:
            self.lstm = nn.LSTM(input_size=self.in_channels, hidden_size=self.hidden, bidirectional=self.bidir, batch_first=True, dropout=self.dropout)

        out_size = self.hidden * 2 if (self.bidir and self.architecture == 'lstm') else self.hidden
        self.classifier = nn.Linear(out_size, self.n_classes)

    def forward(self, x):
        if self.mlp:
            x = torch.transpose(x, 1, 2)
            out = torch.cat([self.avg(c(x)) for c in self.conv], dim=1)
            output = self.mlp(out.flatten())[np.newaxis, :]
        else:
            out, (hn, cn) = self.lstm(x)
            output = out[:, -1]
        return out, (hn, cn), self.classifier(output)
