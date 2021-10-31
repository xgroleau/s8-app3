from typing import Dict

from sklearn.cluster import KMeans
import numpy as np


def kmean_clustering(classes: Dict[str, np.ndarray], n_cluster=5):
    representant = {}
    for key in classes:
        knn_class = KMeans(n_cluster).fit(classes[key]['params'])
        representant[key] = knn_class.cluster_centers_
    return representant
