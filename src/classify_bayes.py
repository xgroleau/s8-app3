import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from src.classifier.bayesian_classifier import BayesianClassifier
from src.classifier.classify import classify
from src.classifier.subclasses import subclass, subclass_param_threshold
from src.images import ImageCollection, export_collection
from src.metrics.fisher_criterion import analyze_fisher_discriminant
from src.params.extract_param import *
from src.params.param import param_nd
import matplotlib.pyplot as plt
import pickle as pkl
from src.visualization import plot_sub_params

from skimage import color as skic

from src.images import rgb_to_cmyk

sys.path.append('../')
CDIR = os.path.dirname(os.path.realpath(__file__))
images_path = os.path.join(CDIR, '../baseDeDonneesImages')

coast = ImageCollection(base_path=images_path, filter_name="coast")
forest = ImageCollection(base_path=images_path, filter_name="forest")
street = ImageCollection(base_path=images_path, filter_name="street")

categorized_collection = {"coast": coast, "forest": forest, "street": street}

param_labels = ['Moyenne Jaune', 'Médiane Bleu', 'Écart-type Vert', 'Moyenne Rouge', 'Moyenne Bleu', 'Moyenne Projection en Rouge', 'Écart-type de la luminosité']

RELOAD_PARAMS = False
if RELOAD_PARAMS:
    params = param_nd(categorized_collection, [(extractor_mean, {'dimension': 2, 'base_function': rgb_to_cmyk}),
                                               (extractor_median, {'dimension': 2, }),
                                               (extractor_std, {'dimension': 1, }),
                                               (extractor_mean, {'dimension': 0, }),
                                               (extractor_mean, {'dimension': 2, }),
                                               (extractor_mean, {'dimension': 2, 'base_function': skic.rgb2yuv}),
                                               (extractor_std, {'dimension': 2, 'base_function': skic.rgb2hsv})
                                               ], num_images=-1)

    f = open("params.pkl", "wb")
    pkl.dump(params, f)
    f.close()
else:
    f = open("params.pkl", "rb")
    params = pkl.load(f)
    f.close()


analyze_fisher_discriminant(params)

bayes = BayesianClassifier(params, bins=1)

view = (3,4)

plot_sub_params(params, view)

params = subclass(params, 'coast', subclass_param_threshold, param_idx=5, threshold=0.05)
plot_sub_params(params, view, param_labels)


bayes2 = BayesianClassifier(params, bins=10)
classify(params, bayes2.fit_multiple, likelihood='arbitrary',  visualize_errors_dims=view)
export_collection({k: v['image_names'] for k, v in params.items()}, "collection.pkl")
analyze_fisher_discriminant(params)

plt.show()
