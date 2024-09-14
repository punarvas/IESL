"""
@filename: Network.py
@authors: Akash Mahajan, Van-Hai Bui
Network architecture configuration and model training
"""

import numpy as np
from probabilistic_demand_prediction.utils.Priors import Prior
from torch.nn import Module, ReLU  # For activation function type restriction
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import os
import json
import time


class NetConfig:
    # Used to initialize model hyperparameters
    def __init__(self, prior: Prior, input_units: np.int32, output_units: np.int32,
                 output_activation: Module, hidden_activation: Module = ReLU(), model_name: str = "Model"):
        self.layer_units = []

        self.input_units = input_units
        self.output_units = output_units
        self.model_name = model_name
        self.prior = prior
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.drop_probability = 0.5  # Special parameters for LSTM
        self.stacked_layers = 2

    def add_layer(self, n_units: np.int32):
        self.layer_units.append(n_units)

    def get_config(self):
        config = {"model_name": self.model_name,
                  "io_units": (self.input_units, self.output_units),
                  "hidden_activation": "ReLU",
                  "prior_name": self.prior.name(),
                  "layer_units": self.layer_units}
        return config


class TrainConfig:
    # Used to initialize training hyperparameters
    def __init__(self, batch_size: np.int32, epochs: np.int32, optimizer: torch.optim.Optimizer,
                 loss: Module, learning_rate: np.float32, save_folder: str):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.999)
        self.save_folder = save_folder

    def get_config(self):
        config = {"batch_size": self.batch_size,
                  "learning_rate": self.learning_rate,
                  "epochs": self.epochs,
                  "save_folder": self.save_folder}
        return config


class TrainHistory:
    def __init__(self):
        self.loss_history = []
        self.nll_history = []
        self.kl_history = []

    def to_file(self, model_name: str, config: TrainConfig):
        if not os.path.exists(model_name):
            os.mkdir(model_name)
        filename = f"{model_name}_{config.epochs}_{config.learning_rate}.json"
        output_data = {"train_loss": self.loss_history,
                       "train_nll": self.nll_history,
                       "train_kl": self.kl_history}
        file_path = os.path.join(model_name, filename)
        with open(file_path, "w") as file:
            json.dump(output_data, file, indent=5)
        print(f"Training history saved to file: {file_path}")


class Trainer:
    def __init__(self, train_loader: DataLoader, validation_loader: DataLoader,
                 model: Module, train_config: TrainConfig):
        self.model = model
        self.train_config = train_config
        self.device = self._training_device()
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.n_batches = len(train_loader)

        self.start_time = None
        self.end_time = None

    def _get_loss(self, output, target):
        try:
            value = self.train_config.loss(output[0], target, output[1]), output[2]
        except IndexError:
            value = self.train_config.loss(output[0], target, output[1])
        return value

    def fit(self):
        model_name = self.model.config.model_name
        print(f"Training model: {model_name}")
        train_history = TrainHistory()
        n_samples = 3
        self.model.to(self.device)
        self.model.train()

        self.start_time = time.time()

        for epoch in range(self.train_config.epochs):
            epoch_loss, epoch_nll, epoch_kl = [], [], []
            for _, (features, target) in enumerate(self.train_loader):
                features = features.to(self.device)
                target = target.to(self.device)
                nll_cum = 0
                kl_cum = 0
                for i in range(n_samples):
                    output = self.model(features)
                    nll_i, kl_i = self._get_loss(output, target)
                    kl_i = kl_i / self.n_batches
                    nll_cum += nll_i
                    kl_cum += kl_i
                nll = nll_cum / n_samples
                kl = kl_cum / n_samples
                loss = nll + kl
                loss.backward()  # self.train_config.loss.backward()
                self.train_config.optimizer.step()
                self.train_config.optimizer.zero_grad()
                epoch_loss.append(loss.item())
                epoch_nll.append(nll.item())
                epoch_kl.append(kl.item())

            mean_loss = np.mean(epoch_loss)
            mean_nll = np.mean(epoch_nll)
            mean_kl = np.mean(epoch_kl)
            train_history.loss_history.append(mean_loss)
            train_history.nll_history.append(mean_nll)
            train_history.kl_history.append(mean_kl)

            self.train_config.lr_scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch: {epoch + 1}, NLL: {mean_nll:.3f}, KL: {mean_kl:.3f}")

        # Save the model

        if not os.path.exists(self.train_config.save_folder):
            os.mkdir(self.train_config.save_folder)

        save_file_name = f"{model_name}_{self.train_config.batch_size}_{self.train_config.epochs}_{self.train_config.learning_rate}.pt"
        save_path = os.path.join(self.train_config.save_folder, save_file_name)
        torch.save(self.model.state_dict(), save_path)

        self.end_time = time.time()
        print(f"Training time: {self.time_in_hours():.2f} hour(s)")

        return train_history

    def evaluate(self, draw_n_samples: np.int32 = 100):
        """
        :param draw_n_samples: How many samples to draw from posterior distribution to generate
        robust mean and variance for each data point
        :return: None

        Evaluates the given model on a validation dataset or a test dataset and exports evaluation
        results in a file which includes target values and sample-wise mean and variance of the predictions
        """
        model_name = self.model.config.model_name
        print(f"Evaluating model: {model_name}")
        self.model.to(self.device)
        self.model.eval()

        samples_mean, samples_variance = [], []
        for _ in range(draw_n_samples):
            y_hat_means, y_hat_variances = [], []  # y_hat means the predictions of the model
            for _, (features, _) in enumerate(self.validation_loader):
                features = features.to(self.device)
                y_hat_mean, y_hat_variance, _ = self.model(features)
                y_hat_means.append(y_hat_mean.detach().cpu().numpy())
                y_hat_variances.append(y_hat_variance.detach().cpu().numpy())
            acc_mean = np.concatenate(y_hat_means, axis=0)
            acc_variance = np.concatenate(y_hat_variances, axis=0)
            samples_mean.append(acc_mean)
            samples_variance.append(acc_variance)

        samples_mean = np.array(samples_mean)
        samples_variance = np.array(samples_variance)

        mixture_mean = np.mean(samples_mean, axis=0)
        mixture_variance = np.mean(samples_variance + np.square(samples_mean), axis=0) - np.square(mixture_mean)
        targets = self.get_targets("validation")

        evaluation_results = {"targets": targets,
                              "mixture_mean": mixture_mean,
                              "mixture_variance": mixture_variance}
        return evaluation_results

    def get_targets(self, set_name: str):
        targets = []
        if set_name == "train":
            for _, (_, target) in enumerate(self.train_loader):
                targets.append(target.numpy())
        elif set_name == "validation":
            for _, (_, target) in enumerate(self.validation_loader):
                targets.append(target.numpy())
        else:
            raise ValueError("Set name is invalid. Choose from [train, validation]")

        return np.concatenate(targets, axis=0)

    def _training_device(self):
        device_name = "cuda:0"
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        device = torch.device(device_name)
        return device

    def time_in_hours(self) -> float:
        return (self.end_time - self.start_time) / 3600


class LSTMTrainer(Trainer):
    def fit(self):
        model_name = self.model.config.model_name
        print(f"Training model: {model_name}")
        train_history = TrainHistory()
        self.model.to(self.device)
        self.model.train()

        self.start_time = time.time()

        for epoch in range(self.train_config.epochs):
            epoch_loss = []
            for _, (features, target) in enumerate(self.train_loader):
                features = features.to(self.device)
                target = target.to(self.device)

                output = self.model(features)
                loss = self._get_loss(output, target)
                loss.backward()
                self.train_config.optimizer.step()
                self.train_config.optimizer.zero_grad()
                epoch_loss.append(loss.item())

            mean_loss = np.mean(epoch_loss)
            train_history.loss_history.append(mean_loss)

            self.train_config.lr_scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch: {epoch + 1}, NLL: {mean_loss:.3f}")

        # Save the model
        if not os.path.exists(self.train_config.save_folder):
            os.mkdir(self.train_config.save_folder)

        save_file_name = f"{model_name}_{self.train_config.batch_size}_{self.train_config.epochs}_{self.train_config.learning_rate}.pt"
        save_path = os.path.join(self.train_config.save_folder, save_file_name)
        torch.save(self.model.state_dict(), save_path)

        self.end_time = time.time()
        print(f"Training time: {self.time_in_hours():.2f} hour(s)")

        return train_history

    def evaluate(self, draw_n_samples: np.int32 = 100):
        model_name = self.model.config.model_name
        print(f"Evaluating model: {model_name}")
        self.model.to(self.device)
        self.model.eval()

        samples_mean, samples_variance = [], []
        for _ in range(draw_n_samples):
            y_hat_means, y_hat_variances = [], []  # y_hat means the predictions of the model
            for _, (features, _) in enumerate(self.validation_loader):
                features = features.to(self.device)
                y_hat_mean, y_hat_variance = self.model(features)
                y_hat_means.append(y_hat_mean.detach().cpu().numpy())
                y_hat_variances.append(y_hat_variance.detach().cpu().numpy())
            acc_mean = np.concatenate(y_hat_means, axis=0)
            acc_variance = np.concatenate(y_hat_variances, axis=0)
            samples_mean.append(acc_mean)
            samples_variance.append(acc_variance)

        samples_mean = np.array(samples_mean)
        samples_variance = np.array(samples_variance)
        mixture_mean = np.mean(samples_mean, axis=0)
        mixture_variance = np.mean(samples_variance + np.square(samples_mean), axis=0) - np.square(mixture_mean)
        targets = self.get_targets("validation")

        evaluation_results = {"targets": targets,
                              "mixture_mean": mixture_mean,
                              "mixture_variance": mixture_variance}
        return evaluation_results
