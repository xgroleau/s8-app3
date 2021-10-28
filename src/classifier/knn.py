from typing import Dict

from sklearn.cluster import KMeans as km
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def knn_classifier(classes: Dict[str, np.ndarray], n_neighbors=1):
    knn_class = KNeighborsClassifier(n_neighbors=1)
    total_length = sum([len(x) for x in classes.values()])
    x = np.zeros((total_length, 2))
    y = np.zeros(len(classes))
    for k, e in classes.items():


    test = np.array([[1, 1], [0, 0]])
    knn_class.fit(test, [[1], [0]])
    predictions = knn_class.predict(test)
