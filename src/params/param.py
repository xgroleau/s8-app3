from typing import Callable, Union, Dict, List, Tuple
import numpy as np
from PIL import Image
from ..images import ImageCollection, random_image_selector
from tqdm import tqdm

ParamExtractor_t = Union[
    Callable[[np.ndarray, any, any], Union[int, float, np.ndarray]],
    Tuple[Callable[[np.ndarray, any, any], Union[int, float, np.ndarray]], Dict]
]


def get_2d_reshaped(arr: np.ndarray):
    """
    Reshape the array to 2d if it's in 1d
    """
    if len(arr.shape) > 1:
        return arr
    else:
        return arr.reshape(-1, 1)


def group_classes(list_classes: List[Dict[str, np.ndarray]]):
    """
    Group the different classes and merge their parameters together
    """
    grouped_classes = {}
    for e_class in list_classes:
        for key in e_class:
            if key in grouped_classes:
                grouped_classes[key] = np.concatenate((grouped_classes[key], get_2d_reshaped(e_class[key])), axis=1)
            else:
                grouped_classes[key] = get_2d_reshaped(e_class[key])

    return grouped_classes


def param_nd(img_coll: Dict[str, ImageCollection],
             param_extraction: List[ParamExtractor_t],
             num_images=200, *args, **kwargs) -> Dict[str, Dict]:
    """
    Extract n params from an image collection for a number of images and returns a dict of the extracted parameters
    """

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


def param_remove_unused(params, to_remove):
    """
    Remove a parameter from a collection of parameters
    """
    to_remove.sort(reverse=True)
    for x in params.keys():
        for index in to_remove:
            params[x]['params'] = np.delete(params[x]['params'], index, axis=1)

    return params
