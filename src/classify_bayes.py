import os
import sys

from skimage.color import rgb2lab

from src.classifier.bayesian_classifier import BayesianClassifier
from src.classifier.classify import classify
from src.classifier.subclasses import subclass, subclass_param_threshold
from src.images import ImageCollection, export_collection, load_collection_from_file
from src.metrics.fisher_criterion import analyze_fisher_discriminant
from src.params.extract_param import *
from src.params.param import param_nd, param_remove_unused, add_params_to_existing
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

param_labels = ["Unique X", "Unique Y", "Unique Z", "Mean R", "Mean G", "Mean, B", "Median R", "Median G", "Median B",
                "Std R", "Std B", "Std G"]

RELOAD_PARAMS = True
if RELOAD_PARAMS:
    params = param_nd(categorized_collection, [(extractor_mean, {'dimension': 1, 'base_function': rgb_to_cmyk}),
                                               (extractor_mean, {'dimension': 2, 'base_function': rgb_to_cmyk}),
                                               (extractor_mean, {'dimension': 3, 'base_function': rgb_to_cmyk}),
                                               (extractor_median, {'dimension': 2, 'base_function': skic.rgb2hsv}),
                                               (extractor_std, {'dimension': 1, 'base_function': rgb_to_cmyk}),
                                               ], num_images=-1)

    f = open("params.pkl", "wb")
    pkl.dump(params, f)
    f.close()
else:
    f = open("params.pkl", "rb")
    params = pkl.load(f)
    f.close()

#params=param_remove_unused(params, [0,1])
bayes = BayesianClassifier(params, bins=1)

#add_params_to_existing(params, images_path, (extract_peak_hsv, {'subset_start': 5, 'dimension': 0}))


# create_confusion_matrix(params, bayes.fit_multiple, display=True, agregate=False, likelihood='gaussian')
# create_confusion_matrix(params, bayes.fit_multiple, display=True, agregate=True, likelihood='gaussian')

view = (1)

#params = subclass(params, 'coast', subclass_param_threshold, param_idx=3, threshold=75)
#params = subclass(params, 'forest', subclass_param_threshold, param_idx=13, threshold=60)

# for k in params.keys():
#     params[k]['params'][:, 0] = (params[k]['params'][:, 0] + 150) % 256
#     params[k]['params'][:, 1] = (params[k]['params'][:, 1] + 150) % 256
#     params[k]['params'][:, 2] = (params[k]['params'][:, 2] + 150) % 256
#     params[k]['params'][:, 12] = (params[k]['params'][:, 12] + 150) % 256
#
# for i in range(15):
#     plot_sub_params(params, i)
#
# params = param_remove_unused(params, [0,1,2,5,6,7,8,9,10,11,14])

# for k, v in params.items():
#     params[k]['params'][:, 0] = (params[k]['params'][:, 0] + 50) % 255

# for i in range(8):
#     plot_sub_params(params, i, param_labels)

plot_sub_params(params, view)
# plot_sub_params(params, 2, param_labels)
# plot_sub_params(params, 3, param_labels)
#params = subclass(params, 'coast', subclass_param_threshold, param_idx=10, threshold=15)
# params = subclass(params, 'street', subclass_param_threshold, param_idx=0, threshold=0.3)
# params = subclass(params, 'forest', subclass_param_threshold, param_idx=0, threshold=0.3)
# params = subclass(params, 'street_0', subclass_param_threshold, param_idx=4, threshold=4000)
# params = subclass(params, 'forest_0', subclass_param_threshold, param_idx=2, threshold=6000)
# params = subclass(params, 'forest_0', subclass_param_threshold, param_idx=2, threshold=50)
# plot_sub_params(params, 3, param_labels)
plot_sub_params(params, view)

bayes2 = BayesianClassifier(params, bins=1)

classify(params, bayes2.fit_multiple, likelihood='gaussian', visualize_errors_dims=view)

export_collection({k: v['image_names'] for k, v in params.items()}, "collection.pkl")

analyze_fisher_discriminant(params)

plt.show()
