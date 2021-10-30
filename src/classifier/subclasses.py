from typing import Callable, Dict, List

import numpy as np
import copy


def subclass_param_threshold(params: np.ndarray, param_idx: int, threshold: float):
    x1 = params[params[:, param_idx] < threshold]
    x2 = params[params[:, param_idx] >= threshold]
    return [x1, x2]


def subclass(params: Dict[str, np.ndarray], base_class: str,
             fun: Callable[[np.ndarray, any, any], List[np.ndarray]], *args, **kwargs) -> Dict[str, np.ndarray]:
    new_params = copy.deepcopy(params)
    new_params.pop(base_class)

    subclasses = fun(params[base_class], *args, **kwargs)
    for i, v in enumerate(subclasses):
        new_params[f'{base_class}_{i}'] = v

    return new_params


def aggregate_subclasses(labels):
    for i in range(len(labels)):
        labels[i] = labels[i].split("_")[0]

    return labels
