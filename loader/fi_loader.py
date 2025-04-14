import numpy as np
import os
from torch.utils.data import Dataset
import torch


def get_raw_dataset(data_dir, method, cf, train):
    """
    There are three methods of normalization: z-score, min-max, and decimal-precision
    """
    if train:
        path = os.path.join(data_dir, method, 'training', f'Train_Dst_NoAuction_{method}_CF_{str(cf)}.txt')
    else:
        path = os.path.join(data_dir, method, 'testing', f'Test_Dst_NoAuction_{method}_CF_{str(cf)}.txt')

    return np.loadtxt(str(path)).T


def split_x_y(data):
    """
    X is input data for the models, Y is the label of 5 classification problems,
    where 1 for up-movement, 2 for stationary condition, 3 is for down-movement
    """
    data_length = 40
    x = data[:, :data_length]
    y = data[:, -5:]
    return x, y


def group_ticks(x, y, T, k):
    """
    x_proc: 3D array that groups ticks into sequences of length T
    y_proc: 1D array that selects the kth classification problem
    """
    [N, D] = x.shape

    # x processing
    x_proc = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        x_proc[i - T] = x[i - T:i, :]

    # y processing
    y_proc = y[T - 1:N]
    y_proc = y_proc[:, k] - 1
    return x_proc, y_proc


class FIDataset(Dataset):
    def __init__(self, data_dir, method='Zscore', cf=1, train=True, T=100, k=4):
        self.method = method
        self.cf = cf
        self.train=True

        x, y = split_x_y(get_raw_dataset(data_dir, method, cf, train))
        x, y = group_ticks(x, y, T, k)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

        self.length = len(y)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]
