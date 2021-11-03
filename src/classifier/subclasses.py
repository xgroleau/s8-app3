from typing import Callable, Dict, List

import numpy as np
import copy


def subclass_param_threshold(params: Dict, param_idx: int, threshold: float):
    """
    Seperate a class in two based on a threshold on a given parameter
    @param params: The dictionary of parameters and image names to subclass
    @param param_idx: The index of the parameter on which to apply the threshold
    @param threshold: The threshold at which to subclass
    @return: A list with two dictionaries one for each subclass
    """

    # Get indexes on each side of the threshold
    idx1 = np.where(params['params'][:, param_idx] < threshold)
    idx2 = np.where(params['params'][:, param_idx] >= threshold)

    # Apply threshold on both the parameters and the image names
    x1 = {
        'image_names': [params['image_names'][i] for i in idx1[0]],
        'params': params['params'][idx1]
    }

    x2 = {
        'image_names': [params['image_names'][i] for i in idx2[0]],
        'params': params['params'][idx2]
    }

    return [x1, x2]


def subclass(params: Dict[str, Dict], base_class: str,
             fun: Callable[[Dict, any, any], List[np.ndarray]], *args, **kwargs) -> Dict[str, Dict]:
    """
    Create two subclasses given a class determined by a callable function
    """
    new_params = copy.deepcopy(params)
    new_params.pop(base_class)

    # Get new subclasses and append to other classes
    subclasses = fun(params[base_class], *args, **kwargs)
    for i, v in enumerate(subclasses):
        new_params[f'{base_class}_{i}'] = v

    return new_params


def aggregate_subclasses(labels):
    """
    Aggregates the labels, takes labels of subclasses and aggregates them in the original classes
    """
    label_tmp = []
    for i in range(len(labels)):
        label_tmp.append(labels[i].split("_")[0])

    return label_tmp
