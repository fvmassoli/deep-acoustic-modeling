import argparse
import numpy as np
import pandas as pd

from model_manager import ModelManager
from utils import get_model_info_from_fname

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


def test(args):

    architecture, hidden_size, bidir, out_features, windows, dropout = get_model_info_from_fname(args.checkpointPath.split('/')[-2])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_manager = ModelManager(n_classes=44,
                                 out_channels=out_features,
                                 hidden=hidden_size,
                                 bidir=bidir,
                                 dropout=dropout,
                                 architecture=architecture,
                                 windows=windows,
                                 load_ckt=args.loadModelCkt,
                                 ckt_path=args.checkpointPath,
                                 device=device)
    model_manager.set_eval_mode()

    test_ds = torch.from_numpy(pd.read_csv(args.csvFilePath).to_numpy()).float()[np.newaxis, :]

    output, loss = model_manager.forward(test_ds, None, nn.CrossEntropyLoss)

    output = output.detach().cpu().numpy()
    exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    output = output.squeeze()
    probs = probs.squeeze()

    file = open(args.outputFile, 'w')
    str_ = 'Most probable class: {} --- with prob: {}\n'.format(idx_to_class[np.argmax(output)], probs[np.argmax(output)])
    file.write(str_)
    for idx, out in enumerate(output):
        str_ = str(out) + '  class  ' + idx_to_class[idx] + '  prob: ' + str(probs[idx]) + '\n'
        file.write(str_)
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Syllables')
    parser.add_argument('-l', '--loadModelCkt', action='store_true', help='Load model ckt (default: false)')
    parser.add_argument('-ck', '--checkpointPath', help='Path to model checkpoint')
    parser.add_argument('-f', '--csvFilePath', help='Path to input csv file')
    parser.add_argument('-o', '--outputFile', help='Output .txt file path')
    args = parser.parse_args()
    test(args)
