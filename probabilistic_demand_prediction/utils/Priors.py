"""
@filename: Priors.py
@authors: Akash Mahajan, Van Hai Bui
Prior distributions for probabilistic learning

Inspired by Abdulmajid-Murad/deep_probabilistic_forecast
GitHub page: https://github.com/Abdulmajid-Murad/deep_probabilistic_forecast
"""

import numpy as np
import torch


class Prior:  # Laplacian prior
    def __init__(self, mean, spread):
        self.mean = mean
        self.spread = spread
        self._name = "Laplacian prior"

    @property
    def name(self):
        return self._name

    def log_likelihood(self, x, do_sum: bool = True):
        if do_sum:
            result = (-np.log(2 * self.spread) - torch.abs(x - self.mean) / self.spread).sum()
        else:
            result = -np.log(2 * self.spread) - torch.abs(x - self.mean) / self.spread
        return result


# Isotropic Gaussian prior
class GaussianPrior(Prior):
    def __init__(self, mean, spread):
        super().__init__(mean, spread)
        self._name = "Laplacian prior"

    def log_likelihood(self, x, do_sum: bool = True):
        constant = -0.5 * np.log(2 * np.pi)  # define terms of Gaussian Density Function
        determinant = -np.log(self.spread)
        inner = (x - self.mean) / self.spread
        distance = -0.5 * (inner ** 2)

        if do_sum:
            result = (constant + determinant + distance).sum()
        else:
            result = constant + determinant + distance
        return result

    def log_likelihood_torch(self, x, do_sum=True):
        constant = -0.5 * np.log(2 * np.pi)
        determinant = -torch.log(self.spread)
        inner = (x - self.mean) / self.spread
        distance = -0.5 * (inner ** 2)
        if do_sum:
            result = (constant + determinant + distance).sum()  # sum over all weights
        else:
            result = (constant + determinant + distance)
        return result
