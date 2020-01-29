import os
import argparse
import numpy as np

from run_manager import RunManager
from data_manager import DataManager
from model_manager import ModelManager
from utils import get_train_test_classes_dict, class_idx_dics, get_train_test_dataframes, get_ckt_name

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(args):
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    train_l, test_l = get_train_test_classes_dict()
    class_to_idx, idx_to_class = class_idx_dics(train_l)
    train_df, val_df, test_df = get_train_test_dataframes(train_l, test_l, class_to_idx)

    data_manager = DataManager(dataframes=dict(train=train_df, val=val_df, test=test_df),
                               names=['train', 'val', 'test'],
                               train_mode=True,
                               batch_sizes=dict(train=1, val=1, test=1),
                               pin_memory=torch.cuda.is_available())
    loaders = data_manager.get_loaders()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_manager = ModelManager(n_classes=len(list(set(train_df['class_'].tolist()))),
                                 architecture=args.architecture,
                                 out_channels=args.outFeatures,
                                 hidden=args.hiddenSize,
                                 bidir=args.bidir,
                                 dropout=args.dropout,
                                 windows=args.windows,
                                 load_ckt=False,
                                 ckt_path=None,
                                 device=device)

    optimizer_args = dict(params=model_manager.get_model_params(), lr=args.learningRate, weight_decay=args.weightDecay)
    optimizer = Adam(**optimizer_args) if args.optimizer == 'adam' else SGD(**optimizer_args)

    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5,
                                  patience=5, verbose=True,
                                  min_lr=1.e-7, threshold=0.01) if args.scheduler else None

    print("Training info:"
          "\n\t Optimizer :    {}"
          "\n\t Learning rate: {}"
          "\n\t Weight decay:  {}".format(args.optimizer, args.learningRate, args.weightDecay))

    run_manager = RunManager(model_manager=model_manager,
                             optimizer=optimizer,
                             loaders=loaders,
                             criterion=nn.CrossEntropyLoss(),
                             device=device)

    best_acc = 0.0
    for epoch in range(args.epochs):
        print("=" * 20, 'At epoch: {}/{}'.format(epoch + 1, args.epochs), "=" * 20)
        print()
        if epoch == 0:
            _ = run_manager.val(test=False)
            print()
        run_manager.train()
        print()
        acc = run_manager.val(test=False)
        if acc > best_acc:
            best_acc = acc
            model_manager.save_best_model(optimizer, best_acc, get_ckt_name(args, args.windows))
        print()
        if scheduler is not None:
            scheduler.step(acc, epoch + 1)

    run_manager.val(test=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Syllables')
    parser.add_argument('-s', '--seed', type=int, default=1331, help='Set random seed (default: 1331)')
    parser.add_argument('-a', '--architecture', choices=('lstm', 'mlp'), default='lstm',
                        help='Set model architecture (default: lstm)')
    parser.add_argument('-bd', '--bidir', action='store_true', help='Set Bidirectional LSTM (default: False)')
    parser.add_argument('-of', '--outFeatures', type=int, default=64,
                        help='Total output features channels. Must be a multiple of windows size (default: 15)')
    parser.add_argument('-w', '--windows', type=int, nargs='+', default=[])
    parser.add_argument('-dp', '--dropout', type=float, default=0.2, help='Set dropout probability (default: 0.2)')
    parser.add_argument('-od', '--outputCheckpointDir', default='model_ckt', help='Best output model dir path')
    parser.add_argument('-hs', '--hiddenSize', type=int, default=100, help='Size of inner layer (default: 100)')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='Set number of epochs (default: 150)')
    parser.add_argument('-lr', '--learningRate', type=float, default=0.01, help='Set learning rate (default: 0.01)')
    parser.add_argument('-wd', '--weightDecay', type=float, default=5.e-4, help='Set optimizer weight decay (default: 5.e-4)')
    parser.add_argument('-ba', '--batchAccumulation', type=int, default=256, help='Set batch accumulation iterations (default: 256)')
    parser.add_argument('-op', '--optimizer', choices=('adam', 'sgd'), default='adam', help='Select the optimizer (default: adam)')
    parser.add_argument('-sc', '--scheduler', action='store_true', help='Use learning rate scheduler (default: False)')
    args = parser.parse_args()
    train(args)
