from itertools import chain, repeat
import random
from typing import Dict

from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def _format_knn(classes: Dict[str, np.ndarray]):
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


class KNNClassifier:
    """
    Class that classify using K nearest  neighbors
    """
    def __init__(self, n_neighbors=5):
        """
        Init the classifier with a given number of neighbors to classify in the KNN algorithm
        """
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.labels = []

    def fit(self, classes: Dict[str, np.ndarray]):
        """
        Fit the classifier given a training set of classes
        """
        x, y = _format_knn(classes)
        self.classifier.fit(x, y)
        self.labels = list(classes.keys())

    def predict(self, parameters, *args, **kwargs):
        """
        Predict given an array of parameters. Return the predictions
        """
        predictions = self.classifier.predict(parameters)
        return [self.labels[int(e)] for e in predictions]

