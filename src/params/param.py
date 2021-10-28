from dataclasses import dataclass
from typing import Callable, Union, Dict, List
import numpy as np

from .map_param import map_param
from ..images import ImageCollection
from ..visualization import plot_1d, plot_3d, plot_2d


def get_2d_reshaped(arr: np.ndarray):
    if len(arr.shape) > 1:
        return arr
    else:
        return arr.reshape(-1, 1)


def group_classes(list_classes: List[Dict[str, np.ndarray]]):
    grouped_classes = {}
    for e_class in list_classes:
        for key in e_class:
            if key in grouped_classes:
                grouped_classes[key] = np.concatenate((grouped_classes[key], get_2d_reshaped(e_class[key])), axis=1)
            else:
                grouped_classes[key] = get_2d_reshaped(e_class[key])

    return grouped_classes


def param_1d(img_coll: Dict[str, ImageCollection], param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
                  num_images=200, bins=100, title="", xlabel="", *args, **kwargs) -> Dict[str, np.ndarray]:
    params = {k: map_param(num_images, v, param_extraction, *args, **kwargs) for k, v in img_coll.items()}
    plot_1d(params, bins, title, xlabel)
    return params


def param_nd(img_coll: Dict[str, ImageCollection], param_extraction: List[Callable[[np.ndarray], Union[int, float, np.ndarray]]],
                  num_images=200, title="", xlabel="x", ylabel="y", zlabel="z", *args, **kwargs) -> Dict[str, np.ndarray]:
    params = []
    for param_fun in param_extraction:
        params.append({k: map_param(num_images, v, param_fun, *args, **kwargs) for k, v in img_coll.items()})

    grouped_classes = group_classes(params)

    if len(param_extraction) == 1:
        plot_1d(grouped_classes, title, xlabel)
    elif len(param_extraction) == 2:
        plot_2d(grouped_classes, title, xlabel, ylabel)
    elif len(param_extraction) == 3:
        plot_3d(grouped_classes, title, xlabel, ylabel, zlabel)
    else:
        print("Plot not supported")

    return grouped_classes


def param_3d(img_coll: Dict[str, ImageCollection],
                  param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
                  num_images=200, title="", xlabel="x", ylabel="y", zlabel="z", *args, **kwargs) -> Dict[str, np.ndarray]:
    params = {k: map_param(num_images, v, param_extraction, 3, *args, **kwargs) for k, v in img_coll.items()}
    plot_3d(params, title, xlabel, ylabel, zlabel)
    return params
