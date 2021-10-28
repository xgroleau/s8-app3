from itertools import chain, repeat
import random
from typing import Dict, List

from sklearn.neighbors import KNeighborsClassifier
import numpy as np


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


def format_knn(classes: Dict[str, np.ndarray]):
    total_length = sum([len(x) for x in classes.values()])
    random_key = random.sample(classes.keys(), 1)[0]
    try:
        dim_param = classes[random_key].shape[1]
    except:
        dim_param = 1

    # Flatten the dict in 2 array, one with values, one with labels
    # For X
    x = np.zeros((total_length, dim_param))
    i = 0
    for val in chain.from_iterable(v for v in classes.values()):
        x[i] = val
        i += 1

    # For Y
    class_dict = {}
    i = 0
    for k in classes:
        class_dict[k] = i
        i += 1
    y = np.fromiter(chain.from_iterable(repeat(class_dict[k], classes[k].shape[0]) for k in classes.keys()), dtype=float)
    return x, y


def knn_classifier(classes: Dict[str, np.ndarray], n_neighbors=5):
    x, y = format_knn(classes)
    knn_class = KNeighborsClassifier(n_neighbors=n_neighbors)
    return knn_class.fit(x, y)
