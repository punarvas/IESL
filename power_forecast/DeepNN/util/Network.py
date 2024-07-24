"""
@filename: Network.py
@authors: Akash Mahajan, Van Hai Bui
Network architecture configuration and model training
"""
import numpy as np
from Priors import Prior
from torch.nn import Module, ReLU  # For activation function type restriction
import torch
from torch.utils.data import DataLoader
import os
import json


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
    def __init__(self, batch_size: np.int32, epochs: np.int32, optimizer: Module,
                 loss: Module, learning_rate: np.float32):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer

    def get_config(self):
        config = {"batch_size": self.batch_size,
                  "learning_rate": self.learning_rate,
                  "epochs": self.epochs}
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

    def _get_loss(self, output, target):
        value = self.train_config.loss(output[0], target, output[1]), output[2]
        return value

    def fit(self):
        model_name = self.model.config.model_name
        print(f"Training model: {model_name}")
        train_history = TrainHistory()
        n_samples = 3
        self.model.to(self.device)
        self.model.train()

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

            if (epoch+1) % 10 == 0:
                print(f"Epoch: {epoch+1}, NNL: {mean_nll:.3f}, KL: {mean_kl:.3f}")

        # Save the model

        if not os.path.exists(model_name):
            os.mkdir(model_name)

        save_file_name = f"{model_name}_{self.train_config.epochs}_{self.train_config.learning_rate}.pt"
        save_path = os.path.join(model_name, save_file_name)
        torch.save(self.model.state_dict(), save_path)
        return train_history

    def _training_device(self):
        device_name = "cuda:0"
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        device = torch.device(device_name)
        return device
