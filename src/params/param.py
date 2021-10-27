from dataclasses import dataclass
from typing import Callable, Union
import numpy as np

from .map_param import map_param
from ..images import CategorizedImageCollection
from ..visualization import plot_1d, plot_3d


def param_1d(img_coll: CategorizedImageCollection, param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
                  num_images=200, bins=100, title="", xlabel="", *args, **kwargs):
    coast_param = map_param(num_images, img_coll.coast, param_extraction, *args, **kwargs)
    forest_param = map_param(num_images, img_coll.forest, param_extraction, *args, **kwargs)
    street_param = map_param(num_images, img_coll.street, param_extraction, *args, **kwargs)

    plot_1d(coast_param, forest_param, street_param, bins, title, xlabel)


def param_3d(img_coll: CategorizedImageCollection,
                  param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
                  num_images=200, title="", xlabel="x", ylabel="y", zlabel="z", *args, **kwargs):

    coast_param = map_param(num_images, img_coll.coast, param_extraction, 3, *args, **kwargs)
    forest_param = map_param(num_images, img_coll.forest, param_extraction, 3, *args, **kwargs)
    street_param = map_param(num_images, img_coll.street, param_extraction, 3, *args, **kwargs)

    plot_3d(coast_param, forest_param, street_param, title, xlabel, ylabel, zlabel)
