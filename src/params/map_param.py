from PIL import Image
from src.images import random_image_selector
from src.images.image_collection import ImageCollection
from typing import Callable, Union
import numpy as np


def map_param(num: int, im_coll: ImageCollection, fun: Callable[[np.ndarray, any, any], Union[int, float, np.ndarray]],
              dim=None, *args, **kargs):

    indexes = random_image_selector(num, im_coll)
    dim_tuple = (len(indexes), )
    if dim:
        dim_tuple = dim_tuple + (dim,)
    val = np.zeros(dim_tuple)
    for i in range(len(indexes)):
        imageRGB = np.array(Image.open(im_coll.image_folder + '\\' + im_coll.image_list[indexes[i]]))
        val[i] = fun(imageRGB, *args, **kargs)
    return val