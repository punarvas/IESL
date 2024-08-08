"""
@filename: Vistools.py
@authors: Akash Mahajan, Van Hai Bui
Visualization tools for regression and model performance

Inspired by Abdulmajid-Murad/deep_probabilistic_forecast
GitHub page: https://github.com/Abdulmajid-Murad/deep_probabilistic_forecast
"""

import matplotlib.pyplot as plt
from Network import TrainHistory
import os


target_folder = "PLOTS"
os.makedirs(target_folder, exist_ok=True)

# Update visualization parameters
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.titlesize': 8,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300
})


def plot_training(train_history: TrainHistory, model_name: str, var_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    # Navy blue - #1F77B4
    # dark orange - #FF7F0E
    # forest green - #2CA02C
    color_code = "#FF7F0E"  # dark orange

    axes[0].plot(train_history.nll_history, color=color_code)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Negative Log-Likelihood Loss")

    axes[1].plot(train_history.kl_history, color=color_code)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL Divergence")

    fig.suptitle(f"Training performance of {model_name} ({var_name})")
    fig.tight_layout()

    filename = os.path.join(target_folder, f"training_{model_name}_{var_name}.png")
    plt.savefig(filename)
    print(f"Image saved to {filename}")
