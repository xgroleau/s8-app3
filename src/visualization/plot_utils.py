from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_1d(params: Dict[str, np.ndarray], bins=100, title="", xlabel="", colors=None):
    plt.figure()
    plt.hist(params.values(), bins, alpha=0.5, label=[*params.keys()])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()


def plot_2d(params: Dict[str, np.ndarray], title="", xlabel="", ylabel=""):
    plt.figure()
    for k, e in params.items():
        plt.scatter(e[:, 0], e[:, 1], alpha=0.5, label=k)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def plot_3d(params: Dict[str, np.ndarray], title="", xlabel="x", ylabel="y", zlabel="z", colors=None):
    plt.figure()
    ax = plt.axes(projection='3d')
    for k, e in params.items():
        ax.scatter(e[:, 0], e[:, 1], e[:, 2], alpha=0.5, label=k)
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
