from dataclasses import dataclass
from typing import Callable, Union
import numpy as np

from random import sample
from src.visualization import images_display
from src.visualization.view_histogram import histogrammes

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

def get_images(img_coll: CategorizedImageCollection,
                  param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
                  num_images=200, title="", xlabel="x", ylabel="y", zlabel="z", *args, **kwargs):

    coast_param = map_param(num_images, img_coll.coast, param_extraction, 3, *args, **kwargs)
    coast_param_in_subclass_1 = sample([i for i, value in enumerate(coast_param) if (value[0] < 100 or value[0] > 200)], 6)
    coast_param_in_subclass_2 = sample([i for i, value in enumerate(coast_param) if (100 < value[0] < 200)], 6)
    images_display(coast_param_in_subclass_1, img_coll.coast)
    images_display(coast_param_in_subclass_2, img_coll.coast)
    histogrammes(coast_param_in_subclass_1, img_coll.coast)
    histogrammes(coast_param_in_subclass_2, img_coll.coast)

    forest_param = map_param(num_images, img_coll.forest, param_extraction, 3, *args, **kwargs)
    forest_param_in_subclass_1 = sample([i for i, value in enumerate(forest_param) if (value[0] < 100)], 6)
    forest_param_in_subclass_2 = sample([i for i, value in enumerate(forest_param) if (100 < value[0])], 6)
    images_display(forest_param_in_subclass_1, img_coll.forest)
    images_display(forest_param_in_subclass_2, img_coll.forest)
    histogrammes(forest_param_in_subclass_1, img_coll.forest)
    histogrammes(forest_param_in_subclass_2, img_coll.forest)

    street_param = map_param(num_images, img_coll.street, param_extraction, 3, *args, **kwargs)
    street_param_in_subclass_1 = sample([i for i, value in enumerate(street_param) if (value[0] < 100 or value[0] > 200)], 6)
    street_param_in_subclass_2 = sample([i for i, value in enumerate(street_param) if (100 < value[0] < 200)], 6)
    images_display(street_param_in_subclass_1, img_coll.street)
    images_display(street_param_in_subclass_2, img_coll.street)
    histogrammes(street_param_in_subclass_1, img_coll.street)
    histogrammes(street_param_in_subclass_2, img_coll.street)

    plot_3d(coast_param, forest_param, street_param, title, xlabel, ylabel, zlabel)
