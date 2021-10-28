import numpy as np
import matplotlib.pyplot as plt


class BayesianClassifier:
    def __init__(self, training_set, bins=100, apriori=None):
        self._training_set_counts = [c_val.shape[0] for c_val in training_set.values()]
        self._total_training_set_counts = np.sum([c.shape[0] for c in training_set.values()])

        self._classes_count = len(training_set)

        self._class_labels = [c_key for c_key in training_set.keys()]

        # Extract apriori probability for each class based on number of samples or user-provided probabilities P(C_i)
        self._apriori = apriori if apriori is not None else self._training_set_counts / self._total_training_set_counts

        # Extract N-dimension probability density for each class P(x|C_i)
        self._probability_density = [np.histogramdd(c_val, bins=bins, density=True) for c_val in
                                     training_set.values()]
        self._probability_density = [{'hist': val[0], 'edges': val[1]} for val in
                                     self._probability_density]

        print('X')

    def fit(self, parameters, cost_matrix=None) -> str:
        if cost_matrix is None:
            cost_matrix = np.ones((self._classes_count, self._classes_count))

        risk = np.zeros(self._classes_count)
        likelihoods = [self._get_likelihood(parameters, i) for i in range(self._classes_count)]

        for i in range(self._classes_count):
            for j in range(self._classes_count):
                if i == j:
                    continue

                risk[i] += cost_matrix[i, j] * likelihoods[j] * self._apriori[j]

        return np.argmin(risk)

    def _get_likelihood(self, parameters, class_idx):
        bin_idx = np.zeros(parameters.shape, dtype=np.int64)
        for i in range(parameters.shape[0]):
            bin_idx[i] = self._probability_density[class_idx]['edges'][i].searchsorted(parameters[i], 'left')

        bin_idx -= 1
        if np.any(bin_idx >= self._probability_density[class_idx]['hist'].shape[0]):
            return 0
        else:
            return self._probability_density[class_idx]['hist'].item(tuple(bin_idx))
