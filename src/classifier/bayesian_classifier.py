import numpy as np
import scipy.stats as stats


class BayesianClassifier:
    def __init__(self, training_set, bins=100, apriori=None):
        self._training_set_counts = [c_val.shape[0] for c_val in training_set.values()]
        self._total_training_set_counts = np.sum([c.shape[0] for c in training_set.values()])

        self._classes_count = len(training_set)

        self._class_labels = [c_key for c_key in training_set.keys()]

        # Extract apriori probability for each class based on number of samples or user-provided probabilities P(C_i)
        self._apriori = apriori if apriori is not None else self._training_set_counts / self._total_training_set_counts

        self._normal_distributions = [
            stats.multivariate_normal(mean=np.mean(c_val, axis=0), cov=np.cov(np.transpose(c_val))) for c_val in
            training_set.values()]

        # Extract N-dimension probability density for each class P(x|C_i)
        self._probability_density = [np.histogramdd(c_val, bins=bins, density=True) for c_val in
                                     training_set.values()]
        self._probability_density = [{'hist': val[0], 'edges': val[1]} for val in
                                     self._probability_density]

    def fit_multiple(self,  parameters, likelihood='arbitrary', cost_matrix=None):
        if len(parameters.shape) == 1:
            parameters = np.expand_dims(parameters, -1)
        return np.array([self.fit(v, likelihood, cost_matrix) for v in parameters])

    def fit(self, parameters, likelihood='arbitrary', cost_matrix=None) -> str:
        if cost_matrix is None:
            cost_matrix = np.ones((self._classes_count, self._classes_count))

        risk = np.zeros(self._classes_count)

        if likelihood == 'gaussian':
            likelihoods = [self._get_gaussian_likelihood(parameters, i) for i in range(self._classes_count)]
        elif likelihood == 'arbitrary':
            likelihoods = [self._get_arbitrary_likelihood(parameters, i) for i in range(self._classes_count)]
        else:
            raise ValueError(f'Likelihood: {likelihood} not supported. Supported values are "gaussian" or "arbitrary"')

        for i in range(self._classes_count):
            for j in range(self._classes_count):
                if i == j:
                    continue

                risk[i] += cost_matrix[i, j] * likelihoods[j] * self._apriori[j]

        return np.argmin(risk)

    def _get_arbitrary_likelihood(self, parameters, class_idx):
        bin_idx = np.zeros(parameters.shape, dtype=np.int64)
        for i in range(parameters.shape[0]):
            bin_idx[i] = self._probability_density[class_idx]['edges'][i].searchsorted(parameters[i], 'left')

        bin_idx -= 1
        if np.any(bin_idx >= self._probability_density[class_idx]['hist'].shape[0]):
            return 0
        else:
            return self._probability_density[class_idx]['hist'].item(tuple(bin_idx))

    def _get_gaussian_likelihood(self, parameters, class_idx):
        return self._normal_distributions[class_idx].pdf(parameters)