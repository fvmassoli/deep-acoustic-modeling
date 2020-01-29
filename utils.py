import os
import pandas as pd
from tqdm import tqdm


DATA_BASE_PATH = ...


def get_train_test_classes_dict():
    """

    :return: dictionary for train and test with
             k: folder name
             v: list of .csv files for the 'k' specific folder
    """

    train = os.path.join(DATA_BASE_PATH, ...)
    test = os.path.join(DATA_BASE_PATH, ...)

    train_l = {}
    test_l = {}
    for idx, d in enumerate([train, test]):
        for dd in os.listdir(os.path.join(d)):
            for syl in os.listdir(os.path.join(d, dd)):

                if '.wav' not in syl:
                    if idx == 0:
                        if dd not in train_l:
                            train_l[dd] = []
                        else:
                            train_l[dd].append(os.path.join(d, dd, syl))
                    else:
                        if dd not in test_l:
                            test_l[dd] = []
                        else:
                            test_l[dd].append(os.path.join(d, dd, syl))

    return train_l, test_l


def class_idx_dics(train_l):
    class_to_idx = {}
    idx_to_class = {}

    l = list(train_l.keys())
    l.sort()

    for idx, syl in enumerate(l):
        class_to_idx[syl] = idx
        idx_to_class[idx] = syl

    return class_to_idx, idx_to_class


def get_train_test_dataframes(train_l, test_l, class_to_idx, test_mode=False):
    """

    :param train_l: dictionary
    :param test_l: dictionary
    :param class_to_idx: dictionary
    :param test_mode: boolean
    :return: dataframe with cols = [path to .csv, class, label]
    """

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    if not test_mode:
        for k in tqdm(train_l.keys(), desc='Creating TRAIN dictionary', leave=False):
            for f in train_l[k]:
                tmp = pd.DataFrame(dict(path=f,
                                        class_=f.split('/')[-2],
                                        label=class_to_idx[f.split('/')[-2]]),
                                   index=[0])
                train_df = train_df.append(tmp, ignore_index=True)

    for k in tqdm(test_l.keys(), desc='Creating TEST dictionary', leave=False):
        for f in test_l[k]:
            tmp = pd.DataFrame(dict(path=f,
                                    class_=f.split('/')[-2],
                                    label=class_to_idx[f.split('/')[-2]]),
                               index=[0])
            test_df = test_df.append(tmp, ignore_index=True)

    if test_mode:
        return test_df

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    len_val_df = int(0.1*len(train_df))

    train_df_ = train_df[:len(train_df)-len_val_df]
    val_df = train_df[len(train_df)-len_val_df:]
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    return train_df_, val_df, test_df


def get_ckt_name(args, windows):
    ws = '_w-'
    for w in windows:
        ws += str(w)+'-'
    ws = ws[:-1]
    if args.architecture == 'lstm':
        ws = '_bd-' + str(args.bidir)
    p = os.path.join('training_output', args.outputCheckpointDir, args.architecture+'_h-' + str(args.hiddenSize)
                     + '_out_channels-' + str(args.outFeatures)
                     + '_lr-' + str(args.learningRate).split('.')[1] + '_ba-' + str(args.batchAccumulation)
                     + '_opt-' + args.optimizer + ws + '_dp-' + str(args.dropout))
    if not os.path.exists(p):
        print('Creating folder: {}'.format(p))
        os.makedirs(p)
    return os.path.join(p, 'best_model.pth')


def get_model_info_from_fname(fname):
    l = fname.split('_')
    architecture = l[0]
    hidden_size = int(l[1].split('-')[1])
    out_features = int(l[3].split('-')[1])
    if architecture == 'lstm':
        bidir = l[-2].split('-')[1]
        bidir = True if bidir == "True" else False
        windows = None
    else:
        bidir = None
        windows = l[-2]
        windows = [int(i) for i in windows.split('-')[1:]]
    dropout = float(l[-1].split('-')[1])
    return architecture, hidden_size, bidir, out_features, windows, dropout
