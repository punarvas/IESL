"""
@filename: Datasets.py
@authors: Akash Mahajan, Van Hai Bui
Preparing time series dataset from numpy arrays of features and targets

Inspired by Abdulmajid-Murad/deep_probabilistic_forecast
GitHub page: https://github.com/Abdulmajid-Murad/deep_probabilistic_forecast
"""

from torch.utils.data import Dataset
import numpy as np
import torch


class TimeSeriesDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, look_back_size: int, sequential: bool):
        self.x = x
        self.y = y
        self.sequential = sequential
        self.look_back_size = look_back_size
        if self.look_back_size == self.x.shape[0]:
            self.indices = [self.look_back_size]
        else:
            self.indices = range(self.look_back_size, self.x.shape[0])   # constant stride = 1

    def __getitem__(self, index):
        start, end = self.get_range(index)
        inputs = self.x[start:end]
        target = self.y[end - 1]

        inputs = torch.from_numpy(inputs).type(torch.float)
        target = torch.from_numpy(target).type(torch.float)

        if not self.sequential:
            inputs = torch.flatten(inputs)

        return inputs, target

    def __len__(self):
        return len(self.indices)

    def get_range(self, index):
        end = self.indices[index]
        start = end - self.look_back_size
        return start, end

    def transform(self):
        # Transform the data
        x_mean = self.x.mean()
        x_std = self.x.std()
        self.x = (self.x - x_mean) / x_std
