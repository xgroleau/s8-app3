from typing import Tuple

import numpy as np
import scipy.stats as stats

from src.visualization import plot_2d


def plot_nd(categorized_params, ellipsis):
    pass


class BayesianClassifier:
    """
    Class that classify using naive bayes classification
    """
    def __init__(self, training_set, bins=100, apriori=None):
        """
        Creates and train a bayesian classifier
        """
        self._training_set_counts = [c_val['params'].shape[0] for c_val in training_set.values()]
        self._total_training_set_counts = np.sum([c['params'].shape[0] for c in training_set.values()])

        self._classes_count = len(training_set)
        self._class_labels = [c_key for c_key in training_set.keys()]
        self._dims = training_set[self._class_labels[0]]['params'].shape[1]

        # Extract apriori probability for each class based on number of samples or user-provided probabilities P(C_i)
        self._apriori = apriori if apriori is not None else self._training_set_counts / self._total_training_set_counts

        self._normal_distributions = [
            stats.multivariate_normal(mean=np.mean(c_val['params'], axis=0), cov=np.cov(np.transpose(c_val['params'])),
                                      allow_singular=True) for c_val in
            training_set.values()]

        # Extract N-dimension probability density for each class P(x|C_i)
        self._probability_density = [np.histogramdd(c_val['params'], bins=bins, density=True) for c_val in
                                     training_set.values()]
        self._probability_density = [{'hist': val[0], 'edges': val[1]} for val in
                                     self._probability_density]

    def fit_multiple(self, parameters, likelihood='arbitrary', cost_matrix=None):
        """
        Fit on a multiple parameters
        Returns an numpy array of the prediction of the classes
        """
        if len(parameters.shape) == 1:
            parameters = np.expand_dims(parameters, -1)
        return np.array([self.fit(v, likelihood, cost_matrix) for v in parameters])

    def fit(self, parameters, likelihood='arbitrary', cost_matrix=None) -> str:
        """
        Fit one parameter
        Returns an numpy array of the prediction of the classes
        """
        if cost_matrix is None:
            cost_matrix = np.ones((self._classes_count, self._classes_count))

        risk = np.zeros(self._classes_count)

        # Precaculate likelihoods for every class (P(x|Ci))
        if likelihood == 'gaussian':
            likelihoods = [self._get_gaussian_likelihood(parameters, i) for i in range(self._classes_count)]
        elif likelihood == 'arbitrary':
            likelihoods = [self._get_arbitrary_likelihood(parameters, i) for i in range(self._classes_count)]
        else:
            raise ValueError(f'Likelihood: {likelihood} not supported. Supported values are "gaussian" or "arbitrary"')

        # Compute the Bayes risk
        for i in range(self._classes_count):
            for j in range(self._classes_count):
                if i == j:
                    continue

                risk[i] += cost_matrix[i, j] * likelihoods[j] * self._apriori[j]

        return self._class_labels[np.argmin(risk)]

    def display_decision_boundary(self, param_indexes: Tuple, likelihood='arbitrary', cost_matrix=None):
        """
        Displays the decision boundary for the given parameters.
        (For best results, use on a classifier that only has the amount of parameters used for visualization)
        """
        num_points = 10000
        test_data = np.zeros((self._dims, num_points))

        # Generate random data spanning the range of the training data
        for i in range(self._dims):
            dim_min = np.min([v['edges'][i][0] for v in self._probability_density])
            dim_max = np.min([v['edges'][i][-1] for v in self._probability_density])
            test_data[i] = (dim_max - dim_min) * np.random.random(num_points) + dim_min

        test_data = test_data.T
        labels = self.fit_multiple(test_data, likelihood, cost_matrix)

        # Categorize the data using the classifier
        categorized_params = {}
        for k in self._class_labels:
            indexes = np.where(labels == k)
            categorized_params[k] = {'params': test_data[indexes]}

        # Plot classified data
        sub_params = {k: {'params': v['params'][:, param_indexes]} for k, v in categorized_params.items()}
        plot_2d(sub_params, ellipsis=False)

    def _get_arbitrary_likelihood(self, parameters, class_idx):
        """
        Returns the likelihood that a set of parameters belongs to a class using a histogram as a lookup
        """
        bin_idx = np.zeros(parameters.shape, dtype=np.int64)
        for i in range(parameters.shape[0]):
            bin_idx[i] = self._probability_density[class_idx]['edges'][i].searchsorted(parameters[i], 'left')

        bin_idx -= 1
        if np.any(bin_idx >= self._probability_density[class_idx]['hist'].shape[0]):
            return 0
        else:
            return self._probability_density[class_idx]['hist'].item(tuple(bin_idx))

    def _get_gaussian_likelihood(self, parameters, class_idx):
        """
        Returns the likelihood that a set of parameters belongs to a class using a multivariate normal distribution
        """
        return self._normal_distributions[class_idx].pdf(parameters)
