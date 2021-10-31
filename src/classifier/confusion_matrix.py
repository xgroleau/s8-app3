from typing import Dict, Callable
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from src.classifier.subclasses import aggregate_subclasses


def create_confusion_matrix(params: Dict[str, Dict], fit_function: Callable[[np.ndarray, any, any], str],
                            display=False,
                            agregate=False,
                            *args,
                            **kwargs):

    labels = [k for k in params.keys()]
    expected_labels = [k for k in params.keys() for _ in range(len(params[k]['params']))]
    flattened_params = np.concatenate([v['params'] for v in params.values()])

    fitted_labels = fit_function(flattened_params, *args, **kwargs)

    if agregate:
        expected_labels = aggregate_subclasses(expected_labels)
        fitted_labels = aggregate_subclasses(fitted_labels)
        # Convert to aggregated version and remove duplicates
        labels = list(dict.fromkeys(aggregate_subclasses(labels)))

    confusion_matrix = metrics.confusion_matrix(expected_labels, fitted_labels, labels=labels, normalize='true')

    if display:
        display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
        display.plot()

    return confusion_matrix
