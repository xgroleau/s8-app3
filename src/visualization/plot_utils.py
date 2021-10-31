from typing import Dict, Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np


def plot_1d(params: Dict[str, Dict], bins=100, title="", xlabel="", colors=None):
    plt.figure()
    plt.hist([v['params'] for v in params.values()], bins, alpha=0.5, label=[*params.keys()])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()


def plot_2d(params: Dict[str, np.ndarray], title="", xlabel="x", ylabel="y", colors=None):
    plt.figure()
    ax = plt.axes()
    for k, e in params.items():
        ax.scatter(e['params'][:, 0], e['params'][:, 1], alpha=0.5, label=k)
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_3d(params: Dict[str, np.ndarray], title="", xlabel="x", ylabel="y", zlabel="z", colors=None):
    plt.figure()
    ax = plt.axes(projection='3d')
    for k, e in params.items():
        ax.scatter(e['params'][:, 0], e['params'][:, 1], e['params'][:, 2], alpha=0.5, label=k)
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()


def plot_sub_params(params: Dict[str, np.ndarray], param_indexes: Union[Tuple, int],
                    param_labels: Union[List[str], None] = None, *args, **kwargs):
    sub_params = {k: {'params': v['params'][:, param_indexes]} for k, v in params.items()}

    if isinstance(param_indexes, int):
        xlabel = param_labels[param_indexes] if param_labels is not None else "x"
        plot_1d(sub_params, xlabel=xlabel, *args, **kwargs)
    elif len(param_indexes) == 2:
        xlabel = param_labels[param_indexes[0]] if param_labels is not None else "x"
        ylabel = param_labels[param_indexes[1]] if param_labels is not None else "y"
        plot_2d(sub_params, xlabel=xlabel, ylabel=ylabel, *args, **kwargs)
    elif len(param_indexes) == 3:
        xlabel = param_labels[param_indexes[0]] if param_labels is not None else "x"
        ylabel = param_labels[param_indexes[1]] if param_labels is not None else "y"
        zlabel = param_labels[param_indexes[2]] if param_labels is not None else "z"
        plot_3d(sub_params, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, *args, **kwargs)
    else:
        raise ValueError(f'Param indexes must contain between 1 and 3 parameters')
