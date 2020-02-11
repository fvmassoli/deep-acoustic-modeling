import os
import argparse

from run_manager import RunManager
from data_manager import DataManager
from model_manager import ModelManager
from utils import get_train_test_classes_dict, class_idx_dics, get_train_test_dataframes, get_model_info_from_fname

from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn


idx_to_class = {0: 'di',
                1: 'dje',
                2: 'do',
                3: 'due',
                4: 'dze',
                5: 'kwa',
                6: 'kwan',
                7: 'kwe',
                8: 'kwin',
                9: 'la',
                10: 'lle',
                11: 'mi',
                12: 'nno',
                13: 'no',
                14: 'o',
                15: 'ran',
                16: 'ro',
                17: 'se',
                18: 'sei',
                19: 'sil',
                20: 'sp',
                21: 'ssan',
                22: 'sse',
                23: 'tSa',
                24: 'tSen',
                25: 'tSi',
                26: 'tSin',
                27: 'tSo',
                28: 'ta',
                29: 'ti',
                30: 'to',
                31: 'tre',
                32: 'tren',
                33: 'ttan',
                34: 'tte',
                35: 'tto',
                36: 'ttor',
                37: 'ttro',
                38: 'tu',
                39: 'u',
                40: 'un',
                41: 'van',
                42: 've',
                43: 'ven'}


def evaluate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=range(len(list(idx_to_class.keys()))))


def show_classification_report(y_true, y_pred):
    target_names = ["Class {}".format(i) for i in range(list(idx_to_class.keys()))]
    print(classification_report(y_true, y_pred, target_names=target_names))


def test(args): 

    output_file = './output_file.txt'

    if not os.path.exists(output_file):
        file_object = open(output_file, 'w')
    else:
        file_object = open(output_file, 'a')

    train_l, test_l = get_train_test_classes_dict()
    class_to_idx, idx_to_class = class_idx_dics(train_l)
    test_df = get_train_test_dataframes(train_l, test_l, class_to_idx, True)

    architecture, hidden_size, bidir, out_features, windows, dropout = get_model_info_from_fname(args.checkpointPath.split('/')[-2])

    print("Chekcpoint inof: {} - {} - {} - {} - {} - {}".format(architecture, hidden_size, bidir, out_features, windows, dropout))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_manager = ModelManager(n_classes=len(list(set(test_df['class_'].tolist()))),
                                 out_channels=out_features,
                                 hidden=hidden_size,
                                 bidir=bidir,
                                 dropout=dropout,
                                 architecture=architecture,
                                 windows=windows,
                                 load_ckt=args.loadModelCkt,
                                 ckt_path=args.checkpointPath,
                                 device=device)

    data_manager = DataManager(dataframes=dict(test=test_df),
                               names=['test'],
                               train_mode=False,
                               batch_sizes=dict(test=1),
                               pin_memory=torch.cuda.is_available())
    loaders = data_manager.get_loaders()

    run_manager = RunManager(model_manager=model_manager,
                             optimizer=None,
                             loaders=loaders,
                             criterion=nn.CrossEntropyLoss(),
                             device=device)

    acc = run_manager.val(test=True)

    str_ = "{} {} {} {}\n".format(hidden_size, out_features, bidir, acc)
    file_object.write(str_)

    file_object.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Syllables')
    parser.add_argument('-l', '--loadModelCkt', action='store_true', help='Load model ckt (default: false)')
    parser.add_argument('-ck', '--checkpointPath', help='Path to model checkpoint')
    args = parser.parse_args()
    test(args)
