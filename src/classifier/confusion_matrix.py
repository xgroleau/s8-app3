from typing import Dict, Callable
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def create_confusion_matrix(params: Dict[str, np.ndarray], fit_function: Callable[[np.ndarray, any, any], str],
                            display=False,
                            *args,
                            **kwargs):

    labels = [k for k in params.keys()]
    expected_labels = [k for k in params.keys() for _ in range(len(params[k]))]
    flattened_params = np.concatenate(list(params.values()))

    fitted_labels = fit_function(flattened_params, *args, **kwargs)

    confusion_matrix = metrics.confusion_matrix(expected_labels, fitted_labels)
    confusion_matrix_normalized = metrics.confusion_matrix(expected_labels, fitted_labels, normalize="true")

    if display:
        display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
        display_normalized = metrics.ConfusionMatrixDisplay(confusion_matrix_normalized, display_labels=labels)

        display.plot()
        display_normalized.plot()


    return confusion_matrix
