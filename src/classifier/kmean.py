from typing import Dict

from sklearn.cluster import KMeans
import numpy as np


def kmean_clustering(classes: Dict[str, np.ndarray], n_cluster=5):
    """
    Returns a number of representant for each classes.
    Every classes gets the same number of representant
    """
    representant = {}
    for key in classes:
        knn_class = KMeans(n_cluster, n_init=50, max_iter=500, tol=1e-5).fit(classes[key]['params'])
        representant[key] = knn_class.cluster_centers_
    return representant
