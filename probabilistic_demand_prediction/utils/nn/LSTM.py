"""
@filename: LSTM.py
@authors: Akash Mahajan, Van-Hai Bui
Preparing time series dataset from numpy arrays of features and targets

Inspired by Abdulmajid-Murad/deep_probabilistic_forecast
GitHub page: https://github.com/Abdulmajid-Murad/deep_probabilistic_forecast
"""

import torch.nn as nn
import torch
from probabilistic_demand_prediction.utils.Network import NetConfig
from torch.autograd import Variable
import torch.nn.functional as func
import numpy as np


class LSTM(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config
        self.device = self._get_device()

        self._layers = nn.ModuleList()
        self.drop_probability = self.config.drop_probability
        self.num_layers = self.config.stacked_layers
        input_units = self.config.input_units

        for hidden_units in self.config.layer_units:
            layer = nn.LSTM(input_size=input_units,
                            hidden_size=hidden_units,
                            num_layers=self.num_layers, batch_first=True)
            self._layers.append(layer)
            input_units = hidden_units

        hidden_units = input_units
        self.fully_connected = nn.Linear(hidden_units, 2 * self.config.output_units)
        self.output_activation = self.config.output_activation

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        lstm_output = x
        for i, hidden_size in enumerate(self.config.layer_units):
            hidden_state = self.init_hidden_state(batch_size, hidden_size)
            output, _ = self._layers[i](lstm_output, hidden_state)
            output = func.dropout(output, p=self.drop_probability, training=True)
            lstm_output = output

        output = lstm_output[:, -1, :]
        output = self.fully_connected(output)
        mean = output[:, :self.config.output_units]
        variance = self.output_activation(output[:, self.config.output_units:]) + 1e-06
        return mean, variance

    def init_hidden_state(self, batch_size, hidden_size):
        hidden_state = Variable(torch.zeros(self.num_layers, batch_size, hidden_size)).to(self.device)
        cell_state = Variable(torch.zeros(self.num_layers, batch_size, hidden_size)).to(self.device)
        return hidden_state, cell_state

    def _get_device(self):
        device_name = "cuda:0"
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        device = torch.device(device_name)
        return device
