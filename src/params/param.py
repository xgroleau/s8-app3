from typing import Callable, Union, Dict, List, Tuple
import numpy as np
from PIL import Image
from ..images import ImageCollection, random_image_selector
from ..visualization import plot_1d
from tqdm import tqdm

ParamExtractor_t = Union[
    Callable[[np.ndarray, any, any], Union[int, float, np.ndarray]],
    Tuple[Callable[[np.ndarray, any, any], Union[int, float, np.ndarray]], Dict]
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
             title="", xlabel="", *args, **kwargs):
    params = {}

    for im_class, im_coll in img_coll.items():
        indexes = random_image_selector(num_images, im_coll)
        image_names = [im_coll.image_list[indexes[i]] for i in range(len(indexes))]

        class_params = np.zeros(len(image_names))

        for i, img in enumerate(image_names):
            rgb = np.array(Image.open(im_coll.image_folder + '\\' + img))

            if isinstance(param_extraction, tuple):
                class_params[i] = param_extraction[0](rgb, *args, **{**param_extraction[1], **kwargs})
            else:
                class_params[i] = param_extraction(rgb, *args, **kwargs)

        params[im_class] = {
            'image_names': image_names,
            'params': class_params
        }

    plot_1d(params, bins, title, xlabel)
    return params


def param_nd(img_coll: Dict[str, ImageCollection],
             param_extraction: List[ParamExtractor_t],
             num_images=200, *args, **kwargs) -> Dict[str, Dict]:

    params = {}

    for im_class, im_coll in img_coll.items():
        indexes = random_image_selector(num_images, im_coll)
        image_names = [im_coll.image_list[indexes[i]] for i in range(len(indexes))]

        class_params = np.zeros((len(image_names), len(param_extraction)))

        for i, img in tqdm(enumerate(image_names), total=len(image_names)):
            rgb = np.array(Image.open(im_coll.image_folder + '\\' + img))

            for j, param_fun in enumerate(param_extraction):
                if isinstance(param_fun, tuple):
                    class_params[i, j] = param_fun[0](rgb, *args, **{**param_fun[1], **kwargs})
                else:
                    class_params[i, j] = param_fun(rgb, *args, **kwargs)

        params[im_class] = {
            'image_names': image_names,
            'params': class_params
        }

    return params
