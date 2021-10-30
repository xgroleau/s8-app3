import os
import sys

from src.classifier.bayesian_classifier import BayesianClassifier
from src.classifier.confusion_matrix import create_confusion_matrix
from src.images import ImageCollection
from src.params.extract_param import *
from src.params.param import param_nd
import matplotlib.pyplot as plt

import pickle as pkl

from src.visualization import plot_sub_params

sys.path.append('../')
CDIR = os.path.dirname(os.path.realpath(__file__))
images_path = os.path.join(CDIR, '../baseDeDonneesImages')

coast = ImageCollection(base_path=images_path, filter_name="coast")
forest = ImageCollection(base_path=images_path, filter_name="forest")
street = ImageCollection(base_path=images_path, filter_name="street")

categorized_collection = {"coast": coast, "forest": forest, "street": street}

param_labels = ["Peak position H [5:]", "Peak stdev H", "Peak Y [5:]", "Peak a [100:150]", "Peak b [100:150]",
                "Peak height C [0:50]", "Peak height M [0:50]", "Mean S"]

# params = param_nd(categorized_collection, [(extract_peak_hsv, {'subset_start': 5, 'dimension': 0}),
#                                            (extract_peak_std_hsv, {'dimension': 0}),
#                                            (extract_peak_cmyk, {'subset_start': 5, 'dimension': 2}),
#                                            (extract_peak_lab, {'subset_start': 100, 'subset_end': 150, 'dimension': 1}),
#                                            (extract_peak_lab, {'subset_start': 100, 'subset_end': 150, 'dimension': 2}),
#                                            (extract_peak_height_cmyk, {'subset_start': 0, 'subset_end': 50, 'dimension': 0}),
#                                            (extract_peak_height_cmyk, {'subset_start': 0, 'subset_end': 50, 'dimension': 1}),
#                                            (extract_mean_hsv, {'dimension': 1}), ], num_images=-1)

# a_file = open("params.pkl", "wb")
# pkl.dump(params, a_file)
# a_file.close()

a_file = open("params.pkl", "rb")
params = pkl.load(a_file)
a_file.close()

bayes = BayesianClassifier(params, bins=1)

create_confusion_matrix(params, bayes.fit_multiple, display=True, likelihood='gaussian')

plot_sub_params(params, (0, 3, 5), param_labels)

plt.show()
