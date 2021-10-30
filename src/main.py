'''
Script for image visualization

'''
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image

from src.classifier.bayesian_classifier import BayesianClassifier
from src.classifier.kmean import kmean_clustering
from src.classifier.knn import knn_classifier
from src.images import random_image_selector
from src.images.image_collection import ImageCollection
from src.params import map_param
from src.params.param import param_3d, param_1d, param_nd
from src.params.extract_param import *
from src.visualization import images_display
from src.visualization.view_histogram import histogrammes

sys.path.append('../')
CDIR = os.path.dirname(os.path.realpath(__file__))
images_path = os.path.join(CDIR, '../baseDeDonneesImages')

def main():
    view_hist = False
    coast = ImageCollection(base_path=images_path, filter_name="coast")
    forest = ImageCollection(base_path=images_path, filter_name="forest")
    street = ImageCollection(base_path=images_path, filter_name="street")

    if view_hist:
        im_list_coast = random_image_selector(6, coast)
        images_display(im_list_coast, coast)
        histogrammes(im_list_coast, coast)

        im_list_forest = random_image_selector(6, forest)
        images_display(im_list_forest, forest)
        histogrammes(im_list_forest, forest)

        im_list_street = random_image_selector(6, street)
        images_display(im_list_street, street)
        histogrammes(im_list_street, street)

    categorized_collection = {"coast": coast, "forest": forest, "street": street}

    param_1d(categorized_collection, (extract_peak_hsv, {'subset_start': 5, 'dimension': 0}))
    param_1d(categorized_collection, extract_peak_std_hsv, dimension=0)
    param_1d(categorized_collection, extract_peak_cmyk, subset_start=5, dimension=2)
    param_1d(categorized_collection, extract_peak_lab, subset_start=100, subset_end=150, dimension=1)
    param_1d(categorized_collection, extract_peak_lab, subset_start=100, subset_end=150, dimension=2)
    param_1d(categorized_collection, extract_peak_height_cmyk, subset_start=0, subset_end=50, dimension=0)
    param_1d(categorized_collection, extract_peak_height_cmyk, subset_start=0, subset_end=50, dimension=1)
    param_1d(categorized_collection, extract_mean_hsv, dimension=1)

    # param_3d(categorized_collection, extract_mean_count_pixel_in_slice, subset_start=0, subset_end=75, num_images=-1, title="RGB Slice sum 0 - 75")
    # param_3d(categorized_collection, extract_mean_count_pixel_in_slice, subset_start=75, subset_end=250, num_images=-1, title="RGB Slice sum 75 - 250")
    # get_images(categorized_collection, extract_param_pixels, num_images=-1, title="HSV params pixel")

    # knn = knn_classifier(params, n_neighbors=5)
    # kmean = kmean_clustering(params, n_cluster=5)
    # bayes = BayesianClassifier(params)
    #
    # saturation_coast = map_param(30, categorized_collection['coast'], extract_mean_saturation)
    # saturation_forest = map_param(30, categorized_collection['forest'], extract_mean_saturation)
    # saturation_street = map_param(30, categorized_collection['street'], extract_mean_saturation)
    #
    # bayes_fit_coasts = bayes.fit_multiple(saturation_coast, likelihood='gaussian')
    # bayes_fit_forests = bayes.fit_multiple(saturation_forest, likelihood='gaussian')
    # bayes_fit_streets = bayes.fit_multiple(saturation_street, likelihood='gaussian')
    #
    # ratio_coasts = np.sum((bayes_fit_coasts == 0)) / len(bayes_fit_coasts)
    # ratio_forest = np.sum((bayes_fit_forests == 1)) / len(bayes_fit_forests)
    # ratio_streets = np.sum((bayes_fit_streets == 2)) / len(bayes_fit_streets)
    #
    # print(f"{ratio_coasts} Arr: {bayes_fit_coasts}")
    # print(f"{ratio_forest} Arr: {bayes_fit_forests}")
    # print(f"{ratio_streets} Arr: {bayes_fit_streets}")

    plt.show()


if __name__ == '__main__':
    main()

