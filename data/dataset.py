import os.path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def data_process(data_list):
    user_list, item_list, unique_user_list = [], [], []
    for data in data_list:
        user_list.append(np.array(data[0]).item())
        item_list.append(np.array(data[1]).item())
    return np.array(user_list), np.array(item_list), np.unique(np.array(user_list)), len(data_list)


class RecDataset(Dataset):
    def __init__(self, features, num_user, num_item, train_mat=None, num_ng=0, is_training=None, model='dmf'):
        super(RecDataset, self).__init__()
        # Note that the labels are only useful when training, we thus add them in the ng_sample() function.
        self.features_ps = features
        self.num_user = num_user
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

        if model == 'lgcn' and self.is_training:
            self.prepare_lgcn()

    def ng_sample(self):
        assert self.is_training == True, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def prepare_lgcn(self):
        self.split = False
        self.folds = 100
        self.graph = None
        self._user, self._item, self._unique_users, self._data_size = data_process(self.features_ps)

        self._user_item_net = sp.csr_matrix((np.ones(len(self._user)), (self._user, self._item)),
                                            shape=(self.num_user, self.num_item))

        self.users_d = np.array(self._user_item_net.sum(axis=1)).squeeze()
        self.users_d[self.users_d == 0.] = 1.
        self.items_d = np.array(self._user_item_net.sum(axis=0)).squeeze()
        self.items_d[self.items_d == 0.] = 1.

        self.all_pos = self.get_user_pos_items(list(range(self.num_user)))

    def get_user_pos_items(self, users):
        postive_items = []
        for user in users:
            postive_items.append(self._user_item_net[user].nonzero()[1])
        return postive_items

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        # return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_user + self.num_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.num_user + self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce())
        return A_fold

    def get_sparse_graph(self, save_path):
        if self.graph is None:
            if os.path.exists(os.path.join(save_path, 's_pre_adj_mat.npz')):
                norm_adj = sp.load_npz(os.path.join(save_path, 's_pre_adj_mat.npz'))
            else:
                adj_mat = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item),
                                        dtype=np.float32)
                adjacency_mat = adj_mat.tolil()
                r = self._user_item_net.tolil()
                adjacency_mat[:self.num_user, self.num_user:] = r
                adjacency_mat[self.num_user:, :self.num_user] = r.T
                adjacency_mat = adjacency_mat.todok()
                row_sum = np.array(adjacency_mat.sum(axis=1))
                row_sum[row_sum == 0.] = 1e-12

                d_inv = np.power(row_sum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adjacency_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                sp.save_npz(os.path.join(save_path, 's_pre_adj_mat.npz'), norm_adj)

            if self.split == True:
                self.graph = self._split_A_hat(norm_adj)
            else:
                self.graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.graph = self.graph.coalesce()

        return self.graph

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label
