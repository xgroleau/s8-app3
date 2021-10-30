from dataclasses import dataclass
from typing import Callable, Union, Dict, List, Tuple
import numpy as np

from random import sample
from src.visualization import images_display
from src.visualization.view_histogram import histogrammes

from .map_param import map_param
from ..images import ImageCollection
from ..visualization import plot_1d, plot_3d

ParamExtractor_t = Union[
    Callable[[np.ndarray], Union[int, float, np.ndarray]],
    Tuple[Callable[[np.ndarray], Union[int, float, np.ndarray]], Dict]
]


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


def param_1d(img_coll: Dict[str, ImageCollection], param_extraction: ParamExtractor_t, num_images=200, bins=100,
             title="", xlabel="", *args, **kwargs) -> Dict[str, np.ndarray]:
    if isinstance(param_extraction, tuple):
        params = {k: map_param(num_images, v, param_extraction[0], *args, **{**param_extraction[1], **kwargs}) for k, v in img_coll.items()}
    else:
        params = {k: map_param(num_images, v, param_extraction, *args, **kwargs) for k, v in img_coll.items()}
    plot_1d(params, bins, title, xlabel)
    return params


def param_nd(img_coll: Dict[str, ImageCollection],
             param_extraction: List[ParamExtractor_t],
             num_images=200, *args, **kwargs) -> Dict[str, np.ndarray]:
    params = []
    for i, param_fun in enumerate(param_extraction):
        print(f'Extracting param #{i}')

        if isinstance(param_fun, tuple):
            params.append({k: map_param(num_images, v, param_fun[0], *args, **{**param_fun[1], **kwargs}) for
                      k, v in img_coll.items()})
        else:
            params.append({k: map_param(num_images, v, param_fun, *args, **kwargs) for k, v in img_coll.items()})

    return group_classes(params)


def param_3d(img_coll: Dict[str, ImageCollection],
             param_extraction: ParamExtractor_t,
             num_images=200, title="", xlabel="x", ylabel="y", zlabel="z", *args, **kwargs) -> Dict[str, np.ndarray]:
    if isinstance(param_extraction, tuple):
        params = {k: map_param(num_images, v, param_extraction[0], *args, **{**param_extraction[1], **kwargs}) for k, v in
                  img_coll.items()}
    else:
        params = {k: map_param(num_images, v, param_extraction, *args, **kwargs) for k, v in img_coll.items()}
    plot_3d(params, title, xlabel, ylabel, zlabel)
    return params

# def get_images(img_coll: CategorizedImageCollection,
#                   param_extraction: Callable[[np.ndarray], Union[int, float, np.ndarray]],
#                   num_images=200, title="", xlabel="x", ylabel="y", zlabel="z", *args, **kwargs):
#
#     coast_param = map_param(num_images, img_coll.coast, param_extraction, 3, *args, **kwargs)
#     coast_param_in_subclass_1 = sample([i for i, value in enumerate(coast_param) if (value[0] < 100 or value[0] > 200)], 6)
#     coast_param_in_subclass_2 = sample([i for i, value in enumerate(coast_param) if (100 < value[0] < 200)], 6)
#     images_display(coast_param_in_subclass_1, img_coll.coast)
#     images_display(coast_param_in_subclass_2, img_coll.coast)
#     histogrammes(coast_param_in_subclass_1, img_coll.coast)
#     histogrammes(coast_param_in_subclass_2, img_coll.coast)
#
#     forest_param = map_param(num_images, img_coll.forest, param_extraction, 3, *args, **kwargs)
#     forest_param_in_subclass_1 = sample([i for i, value in enumerate(forest_param) if (value[0] < 100)], 6)
#     forest_param_in_subclass_2 = sample([i for i, value in enumerate(forest_param) if (100 < value[0])], 6)
#     images_display(forest_param_in_subclass_1, img_coll.forest)
#     images_display(forest_param_in_subclass_2, img_coll.forest)
#     histogrammes(forest_param_in_subclass_1, img_coll.forest)
#     histogrammes(forest_param_in_subclass_2, img_coll.forest)
#
#     street_param = map_param(num_images, img_coll.street, param_extraction, 3, *args, **kwargs)
#     street_param_in_subclass_1 = sample([i for i, value in enumerate(street_param) if (value[0] < 100 or value[0] > 200)], 6)
#     street_param_in_subclass_2 = sample([i for i, value in enumerate(street_param) if (100 < value[0] < 200)], 6)
#     images_display(street_param_in_subclass_1, img_coll.street)
#     images_display(street_param_in_subclass_2, img_coll.street)
#     histogrammes(street_param_in_subclass_1, img_coll.street)
#     histogrammes(street_param_in_subclass_2, img_coll.street)
#
#     plot_3d(coast_param, forest_param, street_param, title, xlabel, ylabel, zlabel)
