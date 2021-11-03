from typing import Dict, Callable, Union, Tuple
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from src.classifier.subclasses import aggregate_subclasses
from src.visualization import plot_sub_params


def classify(params: Dict[str, Dict], fit_function: Callable[[np.ndarray, any, any], str],
             normalize_confusion_matrix="true",
             visualize_errors_dims: Union[Tuple, None] = None,
             *args,
             **kwargs):
    labels = [k for k in params.keys()]
    image_names = [n for k in params.keys() for n in params[k]['image_names']]
    expected_labels = np.array([k for k in params.keys() for _ in range(len(params[k]['params']))])
    flattened_params = np.concatenate([v['params'] for v in params.values()])

    fitted_labels = fit_function(flattened_params, *args, **kwargs)

    display_errors(expected_labels, fitted_labels, image_names)
    display_confusion_matrix(labels, expected_labels, fitted_labels, normalize_confusion_matrix)

    if visualize_errors_dims is not None:
        classified_params = {}
        for k in labels:
            in_class = np.where(fitted_labels == k)
            correctly_classified = np.where(fitted_labels == expected_labels)
            correctly_classified_in_class = np.intersect1d(in_class[0], correctly_classified[0])
            classified_params[k] = {'params': flattened_params[correctly_classified_in_class]}

        classified_params['errors'] = {'params': flattened_params[np.where(fitted_labels != expected_labels)]}

        plot_sub_params(classified_params, visualize_errors_dims)


def display_confusion_matrix(labels, expected_labels, fitted_labels, normalize="true"):
    confusion_matrix = metrics.confusion_matrix(expected_labels, fitted_labels, labels=labels, normalize=normalize)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
    display.plot()

    aggregated_labels = list(dict.fromkeys(aggregate_subclasses(labels)))

    if len(aggregated_labels) < len(labels):
        expected_labels = aggregate_subclasses(expected_labels)
        fitted_labels = aggregate_subclasses(fitted_labels)
        confusion_matrix = metrics.confusion_matrix(expected_labels, fitted_labels, labels=aggregated_labels,
                                                    normalize=normalize)

        display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=aggregated_labels)
        display.plot()


def display_errors(expected_labels, fitted_labels, image_names):
    errors = np.where(fitted_labels != expected_labels)
    print("Wrongly classified images: ")
    for error_idx in errors[0]:
        print(f'{image_names[error_idx]} classified as {fitted_labels[error_idx]}')
