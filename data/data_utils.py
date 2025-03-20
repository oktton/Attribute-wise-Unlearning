import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import DataLoader

from data.dataset import RecDataset


def load_all(args):
    train_rating = args.dataset_path + f'{args.dataset}/preprocess/train_interactions.csv'
    test_negative = args.dataset_path + f'{args.dataset}/preprocess/test_neg_interactions'

    train_data = pd.read_csv(
        train_rating, header=None, names=['user', 'item'], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(test_negative, 'r') as fd:
        line = fd.readline()
        while line is not None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()

    return train_data, test_data, user_num, item_num, train_mat


def prepare_data(args):
    args.logger.info(f'prepare {args.dataset} dataset...')
    train_data, test_data, user_num, item_num, train_mat = load_all(args)
    train_dataset = RecDataset(train_data, user_num, item_num, train_mat, args.num_ng, True, args.model)
    test_dataset = RecDataset(test_data, user_num, item_num, train_mat, 0, False, args.model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)
    return train_loader, test_loader, user_num, item_num


def load_positive_dict(args):
    pos_dict_path = args.dataset_path + f'{args.dataset}/preprocess/pos_dict.npy'
    return np.load(pos_dict_path, allow_pickle=True).item()


def get_label_dict(args):
    file_path = os.path.join(args.dataset_path, f'{args.dataset}/preprocess/user_gender.npy')
    label_dict = np.load(file_path, allow_pickle=True).item()
    return label_dict
