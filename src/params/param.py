from dataclasses import dataclass
from typing import Callable, Union, Dict
import numpy as np

from .map_param import map_param
from ..images import ImageCollection
from ..visualization import plot_1d, plot_3d


def param_1d(img_coll: Dict[str, ImageCollection], param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
                  num_images=200, bins=100, title="", xlabel="", *args, **kwargs):
    params = {k: map_param(num_images, v, param_extraction, *args, **kwargs) for k, v in img_coll.items()}
    plot_1d(params, bins, title, xlabel)


def param_3d(img_coll: Dict[str, ImageCollection],
                  param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
                  num_images=200, title="", xlabel="x", ylabel="y", zlabel="z", *args, **kwargs):
    params = {k: map_param(num_images, v, param_extraction, 3, *args, **kwargs) for k, v in img_coll.items()}
    plot_3d(params, title, xlabel, ylabel, zlabel)
