"""
@filename: Metrics.py
@authors: Akash Mahajan, Van-Hai Bui
Preparing time series dataset from numpy arrays of features and targets

Inspired by Abdulmajid-Murad/deep_probabilistic_forecast
GitHub page: https://github.com/Abdulmajid-Murad/deep_probabilistic_forecast
"""

from scipy import special
import numpy as np

_normcdf = special.ndtr


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.power((y_true - y_pred), 2).mean())


def mean_absolute_error(y_true, y_pred):
    return np.absolute((y_true - y_pred)).mean()


def mean_absolute_percentage_error(y_true, y_pred):
    n = len(y_true)
    return (1/n) * np.sum(np.abs(y_true - y_pred) / y_true) * 100


def prediction_interval_coverage_probability(y_true, y_lower, y_upper):
    k_lower = np.maximum(0, np.where((y_true - y_lower) < 0, 0, 1))
    k_upper = np.maximum(0, np.where((y_upper - y_true) < 0, 0, 1))
    PICP = np.multiply(k_lower, k_upper).mean()
    return PICP


def mean_prediction_interval_width(y_lower, y_upper):
    return (y_upper - y_lower).mean()


def _normpdf(x):
    """Probability density function of a univariate standard Gaussian
    distribution with zero mean and unit variance.
    """
    return 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-(x * x) / 2.0)


def crps_gaussian(x, mu, sig):
    x = np.asarray(x)
    mu = np.asarray(mu)
    sig = np.asarray(sig)
    # standadized x
    sx = (x - mu) / sig
    # some precomputations to speed up the gradient
    pdf = _normpdf(sx)
    cdf = _normcdf(sx)
    pi_inv = 1. / np.sqrt(np.pi)
    # the actual crps
    crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    return crps.mean()


def nll_gaussian(x, mu, sig):
    exponent = -0.5 * (x - mu) ** 2 / sig ** 2
    log_coeff = np.log(sig) - 0.5 * np.log(2 * np.pi)
    return - (log_coeff + exponent).mean()
