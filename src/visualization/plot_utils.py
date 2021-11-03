from typing import Dict, Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.patches import Ellipse


def confidence_ellipse(data, ax, scale=1, facecolor='none', **kwargs):
    '''
    ***Testé seulement sur les données du labo


    Inspiration de la documentation de matplotlib 'Plot a confidence ellipse'

    format données de classe np.array([[],
                                       [],
                                       ...
                                       []]
    ax: axe des figures matplotlib
    scale: Facteur d'échelle de l'ellipse
    facecolor and kwargs: Arguments pour la fonction plot de matplotlib

    scale peut être utilisé comme paramètres pour tracer des ellipses à une équiprobabilité
    autre que 1 écart-type
    '''
    cov = np.cov(np.transpose(data))
    lambdas, vectors = np.linalg.eig(cov)
    moy = np.mean(data, axis=0)
    ellipse = Ellipse((0, 0), width=np.sqrt(lambdas[0]) * scale, height=np.sqrt(lambdas[1]) * scale,
                      facecolor=facecolor, **kwargs)
    angle = np.arctan2(vectors[0][1], vectors[0][0])

    # alligne l'ellipse au bon endroit
    transf = transforms.Affine2D() \
        .rotate(-angle) \
        .translate(moy[0], moy[1])

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def plot_1d(params: Dict[str, Dict], bins=100, title="", xlabel="", colors=None):
    """
    Plot an histogram for a 1d parameter
    """
    plt.figure()
    plt.hist([v['params'] for v in params.values()], bins, alpha=0.5, label=[*params.keys()])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()


def plot_2d(params: Dict[str, np.ndarray], title="", xlabel="x", ylabel="y", ellipsis=True, cluster_center=None,
            colors=None):
    colors = ['orange', 'green', 'blue', 'violet', 'cyan', 'gold', 'brown', 'grey']

    plt.figure()
    ax = plt.axes()
    for i, (k, e) in enumerate(params.items()):
        if k == 'errors':
            ax.scatter(e['params'][:, 0], e['params'][:, 1], color='red', label=k)
        else:
            ax.scatter(e['params'][:, 0], e['params'][:, 1], marker='x', color=colors[i], label=k)
            if ellipsis:
                confidence_ellipse(e['params'], ax, edgecolor=colors[i], scale=1)
                confidence_ellipse(e['params'], ax, edgecolor=colors[i], scale=3)
            if cluster_center:
                ax.scatter(cluster_center[k][:, 0], cluster_center[k][:, 1], alpha=1, marker="o", color=colors[i],
                           label=f"{k}_cc")

    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_3d(params: Dict[str, np.ndarray], title="", xlabel="x", ylabel="y", zlabel="z", cluster_center=None,
            colors=None):
    """
    Plot a 3d scatter for the given parameters
    """
    colors = ['orange', 'green', 'blue', 'violet', 'cyan', 'gold', 'brown', 'grey']

    plt.figure()
    ax = plt.axes(projection='3d')
    for i, (k, e) in enumerate(params.items()):
        if k == 'errors':
            ax.scatter(e['params'][:, 0], e['params'][:, 1], e['params'][:, 2], color='red', label=k)
        else:
            ax.scatter(e['params'][:, 0], e['params'][:, 1], e['params'][:, 2], marker="x", alpha=0.5, color=colors[i],
                       label=k)
            if cluster_center:
                ax.scatter(cluster_center[k][:, 0], cluster_center[k][:, 1], cluster_center[k][:, 2], color=colors[i],
                           label=f"{k}_cc")
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()


def plot_sub_params(params: Dict[str, Dict], param_indexes: Union[Tuple, int],
                    param_labels: Union[List[str], None] = None, cluster_center=None, *args, **kwargs):
    sub_params = {k: {'params': v['params'][:, param_indexes]} for k, v in params.items()}
    sub_center = None
    if cluster_center:
        sub_center = {k: v[:, param_indexes] for k, v in cluster_center.items()}

    if isinstance(param_indexes, int):
        xlabel = param_labels[param_indexes] if param_labels is not None else "x"
        plot_1d(sub_params, xlabel=xlabel, *args, **kwargs)
    elif len(param_indexes) == 2:
        xlabel = param_labels[param_indexes[0]] if param_labels is not None else "x"
        ylabel = param_labels[param_indexes[1]] if param_labels is not None else "y"
        plot_2d(sub_params, xlabel=xlabel, ylabel=ylabel, cluster_center=sub_center, *args, **kwargs)
    elif len(param_indexes) == 3:
        xlabel = param_labels[param_indexes[0]] if param_labels is not None else "x"
        ylabel = param_labels[param_indexes[1]] if param_labels is not None else "y"
        zlabel = param_labels[param_indexes[2]] if param_labels is not None else "z"
        plot_3d(sub_params, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, cluster_center=sub_center, *args, **kwargs)
    else:
        raise ValueError(f'Param indexes must contain between 1 and 3 parameters')
