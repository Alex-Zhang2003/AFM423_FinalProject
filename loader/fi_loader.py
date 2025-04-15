import numpy as np
import os
from torch.utils.data import Dataset
import torch


def get_raw_dataset(data_dir, method, cf, train):
    if train:
        path = os.path.join(data_dir, method, 'training', f'Train_Dst_NoAuction_{method}_CF_{str(cf)}.txt')
    else:
        path = os.path.join(data_dir, method, 'testing', f'Test_Dst_NoAuction_{method}_CF_{str(cf)}.txt')

    return np.loadtxt(str(path))


def split_x_y(data):
    data = data.T
    data_length = 40
    x = data[:, :data_length]
    y = data[:, -5:]
    return x, y


def group_ticks(x, y, T, k):
    [N, D] = x.shape

    x_proc = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        x_proc[i - T] = x[i - T:i, :]

    y_proc = y[T - 1:N]
    y_proc = y_proc[:, k] - 1
    return x_proc, y_proc


def extract_stock(raw_data, stock_idx):
    n_boundaries = 4
    boundaries = np.sort(
        np.argsort(np.abs(np.diff(raw_data[0], prepend=np.inf)))[-n_boundaries - 1:]
    )
    boundaries = np.append(boundaries, [raw_data.shape[1]])
    split_data = tuple(raw_data[:, boundaries[i] : boundaries[i + 1]] for i in range(n_boundaries + 1))
    return split_data[stock_idx]



class FIDataset(Dataset):

    def __init__(self, data_dir, method='Zscore', cf=1, train=True, T=100, k=4):
        self.method = method
        self.cf = cf
        self.train = train
        self.data_dir = data_dir
        self.T = T
        self.k = k

        x, y = self.init_dataset()

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

        self.length = len(y)

    def init_dataset(self):
        x_cat = np.array([])
        y_cat = np.array([])
        for stock in [0, 1, 2, 3, 4]:
            day_data = extract_stock(
                get_raw_dataset(self.data_dir, self.method, self.cf, self.train), stock)
            x, y = split_x_y(day_data)
            x_day, y_day = group_ticks(x, y, self.T, self.k)

            if len(x_cat) == 0 and len(y_cat) == 0:
                x_cat = x_day
                y_cat = y_day
            else:
                x_cat = np.concatenate((x_cat, x_day), axis=0)
                y_cat = np.concatenate((y_cat, y_day), axis=0)
        return x_cat, y_cat

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]
