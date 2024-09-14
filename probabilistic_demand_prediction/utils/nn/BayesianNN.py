"""
@filename: BayesianNN.py
@authors: Akash Mahajan, Van Hai Bui
Bayesian Neural Network for time series forecasting

Inspired by Abdulmajid-Murad/deep_probabilistic_forecast
GitHub page: https://github.com/Abdulmajid-Murad/deep_probabilistic_forecast
"""

import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as func
import numpy as np
from probabilistic_demand_prediction.utils.Priors import Prior, GaussianPrior
from probabilistic_demand_prediction.utils.Network import NetConfig


class BayesianLayer(nn.Module):
    def __init__(self, input_units: np.int32, output_units: np.int32, prior: Prior):
        super().__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.prior = prior

        # Learnable parameters -> Initialisation is set empirically.
        # W and b are weights and biases, respectively.

        self.W_mean = nn.Parameter(torch.Tensor(self.input_units, self.output_units).uniform_(-0.1, 0.1))
        self.W_spread = nn.Parameter(torch.Tensor(self.input_units, self.output_units).uniform_(-3, -2))

        self.b_mean = nn.Parameter(torch.Tensor(self.output_units).uniform_(-0.1, 0.1))
        self.b_spread = nn.Parameter(torch.Tensor(self.output_units).uniform_(-3, -2))

        # For variational inference
        self.lpw = 0
        self.lqw = 0

    def forward(self, x: torch.tensor):
        eps_W = Variable(self.W_mean.data.new(self.W_mean.size()).normal_())
        eps_b = Variable(self.b_mean.data.new(self.b_mean.size()).normal_())

        # sample parameters
        std_w = 1e-6 + func.softplus(self.W_spread, beta=1, threshold=20)
        std_b = 1e-6 + func.softplus(self.b_spread, beta=1, threshold=20)

        weights = self.W_mean + 1 * std_w * eps_W
        bias = self.b_mean + 1 * std_b * eps_b
        output = torch.mm(x, weights) + bias.unsqueeze(0).expand(x.shape[0], -1)  # (batch_size, n_output)

        # isotropic gaussian prior
        prior_w = GaussianPrior(self.W_mean, std_w)
        prior_b = GaussianPrior(self.b_mean, std_b)

        lqw = prior_w.log_likelihood_torch(weights, do_sum=True) + prior_b.log_likelihood_torch(bias, do_sum=True)
        lpw = self.prior.log_likelihood(x=weights) + self.prior.log_likelihood(x=bias)
        # print("output:", output)
        # print("lqw - lpw:", self.lqw - self.lpw)
        return output, lqw - lpw  # lqw - lpw


class BayesianNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config
        self.hidden_activation = self.config.hidden_activation
        self._layers = nn.ModuleList()

        input_units = self.config.input_units
        for hidden_units in self.config.layer_units:
            layer = BayesianLayer(input_units=input_units, output_units=hidden_units,
                                  prior=self.config.prior)
            self._layers.append(layer)
            input_units = hidden_units

        # Output layer
        output_layer = BayesianLayer(input_units=input_units, output_units=2 * self.config.output_units,
                                     prior=self.config.prior)
        self._layers.append(output_layer)
        self.output_activation = self.config.output_activation

    def forward(self, x: torch.tensor):
        kl_divergence = 0  # Total KL Divergence

        for layer in self._layers[:-1]:
            x, divergence = layer(x)
            kl_divergence += divergence
            x = self.hidden_activation(x)

        out, kl = self._layers[-1](x)
        kl_divergence += kl

        mean = out[:, :self.config.output_units]
        variance = self.output_activation(out[:, self.config.output_units:]) + 1e-06
        return mean, variance, kl_divergence
