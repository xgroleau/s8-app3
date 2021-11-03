from typing import List, Dict

import numpy as np


def analyze_fisher_discriminant(params: Dict[str, Dict]):
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    class_labels = [k for k in params.keys()]
    for i in range(len(class_labels)):
        print("*****************")
        print(class_labels[i])
        for j in range(len(class_labels)):
            if i == j:
                continue
            score = compute_fisher_criterion([params[class_labels[i]]['params'], params[class_labels[j]]['params']])
            print(f'{class_labels[j]}: {score}')


def compute_fisher_criterion(params: List[np.ndarray]):
    apriori = np.array([v.shape[0] for v in params])
    apriori = apriori / np.sum(apriori)

    all_params = np.concatenate(params)
    mean_all = np.mean(all_params, axis=0)
    means = np.array([np.mean(v, axis=0) for v in params])
    sigmas = np.array([np.std(v, axis=0) for v in params])

    criterion = np.sum((((means-mean_all)**2).T*apriori).T, axis=0) / np.sum((sigmas.T*apriori).T, axis=0)

    return criterion




#compute_fisher_criterion([np.array([[1,5], [2,5], [1,10]]), np.array([[10,101], [11,155], [8,120]])])