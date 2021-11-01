import os
import sys

from src.classifier.bayesian_classifier import BayesianClassifier
from src.classifier.confusion_matrix import create_confusion_matrix
from src.classifier.subclasses import subclass, subclass_param_threshold
from src.images import ImageCollection, export_collection, load_collection_from_file
from src.metrics.fisher_criterion import analyze_fisher_discriminant
from src.params.extract_param import *
from src.params.param import param_nd, param_remove_unused
import matplotlib.pyplot as plt
import pickle as pkl
from src.visualization import plot_sub_params

from skimage import color as skic

from src.images import rgb_to_cmyk, smooth


sys.path.append('../')
CDIR = os.path.dirname(os.path.realpath(__file__))
images_path = os.path.join(CDIR, '../baseDeDonneesImages')

coast = ImageCollection(base_path=images_path, filter_name="coast")
forest = ImageCollection(base_path=images_path, filter_name="forest")
street = ImageCollection(base_path=images_path, filter_name="street")

categorized_collection = {"coast": coast, "forest": forest, "street": street}

param_labels = ["extractor_unique", "extractor_mean r", "extractor_mean g", "extractor_mean b", "extractor_median r",
                "extractor_median g", "extractor_median b", "Mean CMYK"]

RELOAD_PARAMS = False
if RELOAD_PARAMS:
    params = param_nd(categorized_collection, [(extractor_mean, {'dimension': 1, 'base_function': skic.rgb2xyz}),
                                               (extractor_mean, {'dimension': 2, 'base_function': rgb_to_cmyk}),
                                               (extractor_mean, {'dimension': 0, }),
                                               (extractor_mean, {'dimension': 1, }),
                                               (extractor_mean, {'dimension': 2, }),
                                               (extractor_std, {'dimension': 2, 'base_function': skic.rgb2hsv}),
                                               (extractor_std, {'dimension': 3, 'base_function': rgb_to_cmyk}),
                                               ], num_images=-1)

    f = open("params.pkl", "wb")
    pkl.dump(params, f)
    f.close()
else:
    f = open("params_hsv.pkl", "rb")
    params = pkl.load(f)
    f.close()


# Keep extractor_unique, rgb mean, rgb median peut remove median et unique xyz proposition: hsv 11, xyz 4, cmyk 6, cmyk 15
# CMYK Utile pour differencier coast 4, (6 -7), 9, 12, 13, (15 -9)
# CMYK Utile pour differencier street 5, 6, 8
# xyz forest 3, (4), 5
# hsv 2 (11)

params = param_remove_unused(params, [0,1,2, -1,-2, -3])

# params = param_nd(categorized_collection, [(extract_peak_hsv, {'subset_start': 0, 'dimension': 0}),
#                                            (extract_peak_lab, {'dimension': 2}), ], num_images=-1)


bayes = BayesianClassifier(params, bins=1)

#create_confusion_matrix(params, bayes.fit_multiple, display=True, agregate=False, likelihood='gaussian')
#create_confusion_matrix(params, bayes.fit_multiple, display=True, agregate=True, likelihood='gaussian')

view = (3,4,5)

# for k, v in params.items():
#     params[k]['params'][:, 0] = (params[k]['params'][:, 0] + 50) % 255

# for i in range(8):
#     plot_sub_params(params, i, param_labels)

plot_sub_params(params, view, param_labels)
#plot_sub_params(params, 2, param_labels)
#plot_sub_params(params, 3, param_labels)
# params = subclass(params, 'coast', subclass_param_threshold, param_idx=3, threshold=0.34)
# params = subclass(params, 'forest', subclass_param_threshold, param_idx=3, threshold=0.34)
# params = subclass(params, 'street_0', subclass_param_threshold, param_idx=4, threshold=4000)
# params = subclass(params, 'forest_0', subclass_param_threshold, param_idx=2, threshold=6000)
#params = subclass(params, 'forest_0', subclass_param_threshold, param_idx=2, threshold=50)
#plot_sub_params(params, 3, param_labels)
plot_sub_params(params, view, param_labels)
#plot_sub_params(params, 2, param_labels)

bayes2 = BayesianClassifier(params, bins=10)
create_confusion_matrix(params, bayes2.fit_multiple, display=True, agregate=False, likelihood='arbitrary')
create_confusion_matrix(params, bayes2.fit_multiple, display=True, agregate=True, likelihood='arbitrary')

export_collection({k: v['image_names'] for k, v in params.items()}, "collection.pkl")

analyze_fisher_discriminant(params)

plt.show()
