'''
Script for image visualization

'''
import matplotlib.pyplot as plt

from src.classifier.knn import knn_classifier
from src.images import random_image_selector
from src.images.image_collection import ImageCollection
from src.params.extract_param import extract_var, extract_mean_saturation, extract_rb_correlation, extract_mean_ba, \
    extract_peak_b_minus_a
from src.params.param import param_3d, param_1d
from src.visualization import images_display
from src.visualization.view_histogram import histogrammes


def main():
    view_hist = False
    coast = ImageCollection(base_path=r"baseDeDonneesImages", filter_name="coast")
    forest = ImageCollection(base_path=r"baseDeDonneesImages", filter_name="forest")
    street = ImageCollection(base_path=r"baseDeDonneesImages", filter_name="street")

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

    saturation_param = param_1d(categorized_collection, extract_mean_saturation)
    #param_1d(categorized_collection, extract_peak_b_minus_a)
    var_param = param_3d(categorized_collection, extract_var, title="Variance")

    knn_classifier(var_param, n_neighbors=5)


    plt.show()

if __name__ == '__main__':
    main()

