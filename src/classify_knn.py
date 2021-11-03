import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from src.classifier.subclasses import subclass, subclass_param_threshold
from src.classifier.classify import classify
from src.classifier.kmean import kmean_clustering
from src.classifier.knn import KNNClassifier, plot_knn_performance
from src.images import ImageCollection
from src.params.extract_param import *
from src.params.param import param_nd
import matplotlib.pyplot as plt
import pickle as pkl
from src.visualization import plot_sub_params


RELOAD_PARAMS = False
PLOT_PERF_BY_N_REP = False

sys.path.append('../')
CDIR = os.path.dirname(os.path.realpath(__file__))
images_path = os.path.join(CDIR, '../baseDeDonneesImages')

coast = ImageCollection(base_path=images_path, filter_name="coast")
forest = ImageCollection(base_path=images_path, filter_name="forest")
street = ImageCollection(base_path=images_path, filter_name="street")

categorized_collection = {"coast": coast, "forest": forest, "street": street}

if RELOAD_PARAMS:
    params = param_nd(categorized_collection, [(extractor_mean, {'dimension': 1, 'base_function': skic.rgb2xyz}),
                                               (extractor_mean, {'dimension': 2, 'base_function': rgb_to_cmyk}),
                                               (extractor_median, {'dimension': 2, }),
                                               (extractor_std, {'dimension': 1, }),
                                               (extractor_mean, {'dimension': 0, }),
                                               (extractor_mean, {'dimension': 1, }),
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

a_file = open("params.pkl", "rb")
params = pkl.load(a_file)
a_file.close()

param_labels = ["Peak position H [5:]", "Peak stdev H", "Peak Y [5:]", "Peak a [100:150]", "Peak b [100:150]",
                "Peak height C [0:50]", "Peak height M [0:50]", "Mean S"]

view_dims = (3,4,5)

plot_sub_params(params, (0, 1, 2), param_labels)
plot_sub_params(params, (3, 4, 5), param_labels)
plot_sub_params(params, (6, 7), param_labels)

#params = subclass(params, 'coast', subclass_param_threshold, param_idx=0, threshold=75)
#params = subclass(params, 'street', subclass_param_threshold, param_idx=0, threshold=75)
#params = subclass(params, 'forest', subclass_param_threshold, param_idx=0, threshold=100)

if PLOT_PERF_BY_N_REP:
    plot_knn_performance(params, 5, 200, save_path="../figures/knn-performance")

class_representant = kmean_clustering(params, n_cluster=30)
kNN = KNNClassifier(n_neighbors=1)
kNN.fit(class_representant)
plot_sub_params(params, view_dims, param_labels, cluster_center=class_representant, title="012")

classify(params, kNN.predict, normalize_confusion_matrix="true", visualize_errors_dims=view_dims)

plt.show()
