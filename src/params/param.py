from dataclasses import dataclass
from typing import Callable, Union
import numpy as np

from .map_param import map_param
from ..images import CategorizedImageCollection
from ..visualization import plot_1d, plot_3d


def param_1d(img_coll: CategorizedImageCollection, param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
                  num_images=200, bins=100, title="", xlabel=""):
    coast_param = map_param(num_images, img_coll.coast, param_extraction)
    forest_param = map_param(num_images, img_coll.forest, param_extraction)
    street_param = map_param(num_images, img_coll.street, param_extraction)

    plot_1d(coast_param, forest_param, street_param, bins, title, xlabel)


def param_3d(img_coll: CategorizedImageCollection,
                  param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
                  num_images=200, bins=100, title="", xlabel="x", ylabel="y", zlabel="z"):
    coast_param = map_param(num_images, img_coll.coast, param_extraction)
    forest_param = map_param(num_images, img_coll.forest, param_extraction)
    street_param = map_param(num_images, img_coll.street, param_extraction)

    plot_3d(coast_param, forest_param, street_param, bins, title, xlabel, ylabel, zlabel)
